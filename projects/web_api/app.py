import json
import os
from base64 import b64encode
from glob import glob
from io import StringIO
import tempfile
from typing import Tuple, Union
import requests
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

from magic_pdf.data.read_api import read_local_images, read_local_office
import magic_pdf.model as model_config
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.libs.config_reader import get_bucket_name, get_s3_config
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from fastapi import Form

model_config.__use_inside_model__ = True

app = FastAPI()

# Define base paths for downloads and outputs
DOWNLOAD_DIR = Path("downloaded_pdfs")
OUTPUT_DIR_BASE = Path("output_markdown")

# Create these directories if they don't exist
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]

class MemoryDataWriter(DataWriter):
    def __init__(self):
        self.buffer = StringIO()

    def write(self, path: str, data: bytes) -> None:
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            self.buffer.write(data.decode("utf-8"))

    def write_string(self, path: str, data: str) -> None:
        self.buffer.write(data)

    def get_value(self) -> str:
        return self.buffer.getvalue()

    def close(self):
        self.buffer.close()


def init_writers(
    file_path: str = None,
    file: UploadFile = None,
    output_path: str = None,
    output_image_path: str = None,
) -> Tuple[
    Union[S3DataWriter, FileBasedDataWriter],
    Union[S3DataWriter, FileBasedDataWriter],
    bytes,
]:
    """
    Initialize writers based on path type

    Args:
        file_path: file path (local path or S3 path)
        file: Uploaded file object
        output_path: Output directory path
        output_image_path: Image output directory path

    Returns:
        Tuple[writer, image_writer, file_bytes]: Returns initialized writer tuple and file content
    """
    file_extension:str = None
    if file_path:
        is_s3_path = file_path.startswith("s3://")
        if is_s3_path:
            bucket = get_bucket_name(file_path)
            ak, sk, endpoint = get_s3_config(bucket)

            writer = S3DataWriter(
                output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            image_writer = S3DataWriter(
                output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            # 临时创建reader读取文件内容
            temp_reader = S3DataReader(
                "", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            file_bytes = temp_reader.read(file_path)
            file_extension = os.path.splitext(file_path)[1]
        else:
            writer = FileBasedDataWriter(output_path)
            image_writer = FileBasedDataWriter(output_image_path)
            os.makedirs(output_image_path, exist_ok=True)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_extension = os.path.splitext(file_path)[1]
    else:
        # 处理上传的文件
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1]

        writer = FileBasedDataWriter(output_path)
        image_writer = FileBasedDataWriter(output_image_path)
        os.makedirs(output_image_path, exist_ok=True)

    return writer, image_writer, file_bytes, file_extension


def process_file(
    file_bytes: bytes,
    file_extension: str,
    parse_method: str,
    image_writer: Union[S3DataWriter, FileBasedDataWriter],
) -> Tuple[InferenceResult, PipeResult]:
    """
    Process PDF file content

    Args:
        file_bytes: Binary content of file
        file_extension: file extension
        parse_method: Parse method ('ocr', 'txt', 'auto')
        image_writer: Image writer

    Returns:
        Tuple[InferenceResult, PipeResult]: Returns inference result and pipeline result
    """

    ds: Union[PymuDocDataset, ImageDataset] = None
    if file_extension in pdf_extensions:
        ds = PymuDocDataset(file_bytes)
    elif file_extension in office_extensions:
        # 需要使用office解析
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_office(temp_dir)[0]
    elif file_extension in image_extensions:
        # 需要使用ocr解析
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_images(temp_dir)[0]
    infer_result: InferenceResult = None
    pipe_result: PipeResult = None

    if parse_method == "ocr":
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    elif parse_method == "txt":
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    else:  # auto
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

    return infer_result, pipe_result


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


@app.post(
    "/file_parse",
    tags=["projects"],
    summary="Parse files (supports local files and S3)",
)
async def file_parse(
    file: UploadFile = None,
    file_path: str = Form(None),
    parse_method: str = Form("auto"),
    is_json_md_dump: bool = Form(False),
    output_dir: str = Form("output"),
    return_layout: bool = Form(False),
    return_info: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
):
    """
    Execute the process of converting PDF to JSON and MD, outputting MD and JSON files
    to the specified directory.

    Args:
        file: The PDF file to be parsed. Must not be specified together with
            `file_path`
        file_path: The path to the PDF file to be parsed. Must not be specified together
            with `file`
        parse_method: Parsing method, can be auto, ocr, or txt. Default is auto. If
            results are not satisfactory, try ocr
        is_json_md_dump: Whether to write parsed data to .json and .md files. Default
            to False. Different stages of data will be written to different .json files
            (3 in total), md content will be saved to .md file
        output_dir: Output directory for results. A folder named after the PDF file
            will be created to store all results
        return_layout: Whether to return parsed PDF layout. Default to False
        return_info: Whether to return parsed PDF info. Default to False
        return_content_list: Whether to return parsed PDF content list. Default to False
    """
    try:
        if (file is None and file_path is None) or (
            file is not None and file_path is not None
        ):
            return JSONResponse(
                content={"error": "Must provide either file or file_path"},
                status_code=400,
            )

        # Get PDF filename
        file_name = os.path.basename(file_path if file_path else file.filename).split(
            "."
        )[0]
        output_path = f"{output_dir}/{file_name}"
        output_image_path = f"{output_path}/images"

        # Initialize readers/writers and get PDF content
        writer, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=file,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        # Process PDF
        infer_result, pipe_result = process_file(file_bytes, file_extension, parse_method, image_writer)

        # Use MemoryDataWriter to get results
        content_list_writer = MemoryDataWriter()
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()

        # Use PipeResult's dump method to get data
        pipe_result.dump_content_list(content_list_writer, "", "images")
        pipe_result.dump_md(md_content_writer, "", "images")
        pipe_result.dump_middle_json(middle_json_writer, "")

        # Get content
        content_list = json.loads(content_list_writer.get_value())
        md_content = md_content_writer.get_value()
        middle_json = json.loads(middle_json_writer.get_value())
        model_json = infer_result.get_infer_res()

        # If results need to be saved
        if is_json_md_dump:
            writer.write_string(
                f"{file_name}_content_list.json", content_list_writer.get_value()
            )
            writer.write_string(f"{file_name}.md", md_content)
            writer.write_string(
                f"{file_name}_middle.json", middle_json_writer.get_value()
            )
            writer.write_string(
                f"{file_name}_model.json",
                json.dumps(model_json, indent=4, ensure_ascii=False),
            )
            # Save visualization results
            pipe_result.draw_layout(os.path.join(output_path, f"{file_name}_layout.pdf"))
            pipe_result.draw_span(os.path.join(output_path, f"{file_name}_spans.pdf"))
            pipe_result.draw_line_sort(
                os.path.join(output_path, f"{file_name}_line_sort.pdf")
            )
            infer_result.draw_model(os.path.join(output_path, f"{file_name}_model.pdf"))

        # Build return data
        data = {}
        if return_layout:
            data["layout"] = model_json
        if return_info:
            data["info"] = middle_json
        if return_content_list:
            data["content_list"] = content_list
        if return_images:
            image_paths = glob(f"{output_image_path}/*.jpg")
            data["images"] = {
                os.path.basename(
                    image_path
                ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                for image_path in image_paths
            }
        data["md_content"] = md_content  # md_content is always returned

        # Clean up memory writers
        content_list_writer.close()
        md_content_writer.close()
        middle_json_writer.close()

        return JSONResponse(data, status_code=200)

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


from pydantic import BaseModel

class ParseRequest(BaseModel):
    pdf_url: str
    parse_method: str = "auto"

@app.post(
    "/parse_pdf_from_url",
    tags=["projects"],
    summary="Download PDF from URL and parse it",
)
async def parse_pdf_from_url(request: ParseRequest):
    try:
        pdf_url = request.pdf_url
        parse_method = request.parse_method

        # Validate URL (basic check)
        if not pdf_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL scheme. Must be http or https.")

        # Generate filename from URL
        try:
            url_path = Path(pdf_url.split("?")[0]) # Ignore query params for filename
            filename = url_path.name
            if not filename:
                raise ValueError("Could not determine filename from URL")
            # Sanitize filename (basic sanitization)
            filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
            if not any(filename.lower().endswith(ext) for ext in pdf_extensions + office_extensions + image_extensions):
                # If no extension or unknown, try to infer or default
                # For now, let's assume it's a PDF if no clear extension is found in URL path
                # A more robust solution would check Content-Type header after download
                filename = filename + ".pdf"


        except Exception as e:
            logger.error(f"Error generating filename from URL: {pdf_url} - {e}")
            raise HTTPException(status_code=400, detail=f"Could not determine a valid filename from URL: {pdf_url}")


        download_path = DOWNLOAD_DIR / filename
        filename_without_ext = Path(filename).stem
        
        output_path = OUTPUT_DIR_BASE / filename_without_ext
        output_image_path = output_path / "images"

        # Download the PDF
        logger.info(f"Downloading PDF from {pdf_url} to {download_path}")
        try:
            response = requests.get(pdf_url, stream=True, timeout=30) # Added timeout
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download PDF from URL: {e}")

        # Read the downloaded file bytes
        with open(download_path, "rb") as f:
            file_bytes = f.read()
        
        file_extension = os.path.splitext(filename)[1].lower()

        if not file_extension in pdf_extensions + office_extensions + image_extensions:
             # Attempt to get content type if extension is missing or ambiguous
            content_type = response.headers.get('Content-Type', '').lower()
            logger.info(f"File extension not in known list. Content-Type: {content_type}")
            if 'pdf' in content_type:
                file_extension = '.pdf'
                download_path = download_path.rename(download_path.with_suffix('.pdf'))
                filename = filename + ".pdf"
            elif 'officedocument' in content_type or 'presentationml' in content_type or 'wordprocessingml' in content_type:
                # More specific checks can be added here based on typical office MIME types
                # For now, this is a generic catch. Defaulting to .docx if unsure, or could error out.
                # file_extension = '.docx' # Example
                # download_path = download_path.rename(download_path.with_suffix('.docx'))
                # filename = filename + ".docx"
                # For now, let's raise an error if we can't determine office type clearly
                 raise HTTPException(status_code=400, detail=f"Downloaded file from {pdf_url} has an ambiguous or unsupported office content type: {content_type}. Please ensure URL points to a standard PDF, Word, PowerPoint or Image file.")
            elif 'image' in content_type:
                if 'jpeg' in content_type or 'jpg' in content_type:
                    file_extension = '.jpg'
                elif 'png' in content_type:
                    file_extension = '.png'
                else:
                    raise HTTPException(status_code=400, detail=f"Downloaded file from {pdf_url} is an unsupported image type: {content_type}")
                download_path = download_path.rename(download_path.with_suffix(file_extension))
                filename = filename + file_extension
            else:
                os.remove(download_path) # Clean up downloaded file
                raise HTTPException(status_code=400, detail=f"Downloaded file from {pdf_url} does not have a recognized extension and its content type ('{content_type}') is not supported. Please ensure the URL points to a PDF, Office, or image file.")


        # Initialize writers
        # Note: init_writers expects either file_path (for local/S3) or file (UploadFile)
        # Since we've downloaded the file, we can use the file_path argument.
        # However, init_writers also reads the file bytes itself if given a file_path.
        # We already have file_bytes, so we might need to adjust or ensure it works as expected.
        # For now, we'll pass file_path and let it re-read, or adapt init_writers if this is inefficient.
        # A cleaner way would be to modify init_writers or have a new function that accepts file_bytes directly
        # for this use case.
        # Let's try to use the existing init_writers by providing the download_path.

        writer, image_writer, _ , _ = init_writers( # file_bytes and file_extension are re-derived by init_writers
            file_path=str(download_path), # init_writers expects str for local paths
            output_path=str(output_path),
            output_image_path=str(output_image_path),
        )
        
        # Re-read file_bytes as init_writers might not return it when file_path is given
        # or ensure the instance of init_writers correctly loads it.
        # For safety, re-reading here:
        with open(download_path, "rb") as f:
            file_bytes_for_process = f.read()
        
        # file_extension was already determined
        
        # Process PDF
        logger.info(f"Processing downloaded file: {download_path} with parse_method: {parse_method}")
        infer_result, pipe_result = process_file(
            file_bytes=file_bytes_for_process,
            file_extension=file_extension,
            parse_method=parse_method,
            image_writer=image_writer,
        )

        # Save the MD content to a file
        md_file_name = f"{filename_without_ext}.md"
        md_file_path = output_path / md_file_name
        
        # Ensure output_path (which is now a directory per file) exists
        output_path.mkdir(parents=True, exist_ok=True)

        md_content_writer = MemoryDataWriter()
        pipe_result.dump_md(md_content_writer, "", "images") # relative image path
        md_content = md_content_writer.get_value()
        
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Markdown content saved to {md_file_path}")

        md_content_writer.close()
        
        # Also save other JSON outputs if is_json_md_dump behavior is desired (currently not a param for this endpoint)
        # For example:
        # content_list_writer = MemoryDataWriter()
        # pipe_result.dump_content_list(content_list_writer, "", "images")
        # with open(output_path / f"{filename_without_ext}_content_list.json", "w", encoding="utf-8") as f:
        #     f.write(content_list_writer.get_value())
        # content_list_writer.close()


        return JSONResponse(
            content={
                "message": "File processed successfully from URL.",
                "downloaded_pdf_path": str(download_path),
                "markdown_file_path": str(md_file_path),
                "md_content": md_content, # Optionally return md_content directly
                # Add other relevant info from pipe_result or infer_result if needed
            },
            status_code=200,
        )

    except HTTPException as he:
        logger.error(f"HTTPException in /parse_pdf_from_url: {he.detail}")
        raise he # Re-raise HTTPException to let FastAPI handle it
    except Exception as e:
        logger.exception(f"Unexpected error in /parse_pdf_from_url: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

@app.get(
    "/get_pdf/{filename:path}",
    tags=["projects"],
    summary="Serve a downloaded PDF file",
)
async def get_pdf(filename: str):
    try:
        # Construct the full path to the PDF file
        pdf_path = DOWNLOAD_DIR / filename

        # Check if the file exists
        if not pdf_path.is_file():
            logger.error(f"PDF file not found: {pdf_path}")
            raise HTTPException(status_code=404, detail="PDF not found")

        # Return the file as a response
        # The media_type 'application/pdf' tells the browser to display the PDF if possible,
        # or download it. 'filename' suggests the name for the downloaded file.
        return FileResponse(
            path=str(pdf_path),
            media_type='application/pdf',
            filename=filename
        )

    except HTTPException as he:
        # Re-raise HTTPException to let FastAPI handle it
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error in /get_pdf/{filename}: {e}")
        # Return a generic 500 error for other issues
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while trying to serve the PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
