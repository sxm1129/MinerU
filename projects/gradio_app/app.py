# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import re
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pymupdf
from gradio_pdf import PDF
from loguru import logger

from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.tools.common import do_parse, prepare_env


def read_fn(path):
    disk_rw = FileBasedDataReader(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path))


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{str(Path(doc_path).stem)}_{time.time()}'
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = 'ocr'
        else:
            parse_method = 'auto'
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language, logs=None, log_stream=None):
    if logs is None:
        logs = []
    if log_stream is None:
        import io
        log_stream = io.StringIO()
    logs.append("开始转换 PDF ...")
    # tqdm 进度条强制输出到 log_stream
    import sys
    from tqdm import tqdm as tqdm_orig
    def tqdm(*args, **kwargs):
        kwargs['file'] = log_stream
        return tqdm_orig(*args, **kwargs)
    # monkey patch tqdm
    import builtins
    builtins.tqdm = tqdm
    file_path = to_pdf(file_path)
    logs.append("PDF 格式检查完成 ...")
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1, is_ocr,
                                        layout_mode, formula_enable, table_enable, language)
    logs.append(f"PDF 解析完成，输出目录: {local_md_dir}")
    archive_zip_path = os.path.join('./output', compute_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logs.append('压缩成功')
        logger.info('压缩成功')
    else:
        logs.append('压缩失败')
        logger.error('压缩失败')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
    # 检查 PDF 是否存在
    log_stream.flush()
    if not os.path.exists(new_pdf_path):
        logs.append('Error: PDF preview not generated.')
        return 'Error: PDF preview not generated.', txt_content, archive_zip_path, None, logs
    return md_content, txt_content, archive_zip_path, new_pdf_path, logs


latex_delimiters = [{'left': '$$', 'right': '$$', 'display': True},
                    {'left': '$', 'right': '$', 'display': False}]


def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)  # noqa: F841
        logger.info('txt_model init final')
        ocr_model = model_manager.get_model(True, False)  # noqa: F841
        logger.info('ocr_model init final')
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


model_init = init_model()
logger.info(f'model_init: {model_init}')


with open('header.html', 'r') as file:
    header = file.read()


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',  # noqa: E126
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',  # noqa: E126
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',  # noqa: E126
        'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']
add_lang = ['latin', 'arabic', 'cyrillic', 'devanagari']

# all_lang = ['', 'auto']
all_lang = []
# all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])
all_lang.extend([*other_lang, *add_lang])


def to_pdf(file_path):
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 将pdfbytes 写入到uuid.pdf中
            # 生成唯一的文件名
            unique_filename = f'{uuid.uuid4()}.pdf'

            # 构建完整的文件路径
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

            # 将字节数据写入文件
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)

            return tmp_file_path


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                file = gr.File(label='Please upload a PDF or image', file_types=['.pdf', '.png', '.jpeg', '.jpg'])
                url_input = gr.Textbox(label='Or enter PDF URL (e.g. https://arxiv.org/pdf/2409.16040)', placeholder='Enter PDF URL here')
                max_pages = gr.Slider(1, 200, 10, step=1, label='Max convert pages')
                with gr.Row():
                    layout_mode = gr.Dropdown(['doclayout_yolo'], label='Layout model', value='doclayout_yolo')
                    language = gr.Dropdown(all_lang, label='Language', value='ch')
                with gr.Row():
                    formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                    is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                    table_enable = gr.Checkbox(label='Enable table recognition(test)', value=True)
                with gr.Row():
                    change_bu = gr.Button('Convert')
                    clear_bu = gr.ClearButton(value='Clear')
                pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
                log_box = gr.Textbox(label='Process Log', lines=20, interactive=False)
                with gr.Accordion('Examples:'):
                    example_root = os.path.join(os.path.dirname(__file__), 'examples')
                    gr.Examples(
                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                  _.endswith('pdf')],
                        inputs=file
                    )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label='convert result', interactive=False)
                with gr.Tabs():
                    with gr.Tab('Markdown rendering'):
                        md = gr.Markdown(label='Markdown rendering', height=1100, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    with gr.Tab('Markdown text'):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)
        file.change(fn=to_pdf, inputs=file, outputs=pdf_show)

        import requests
        from urllib.parse import urlparse

        import io
        import time
        import contextlib
        import sys
        def redirect_stdouterr(to_stream):
            @contextlib.contextmanager
            def _redirect():
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = to_stream, to_stream
                try:
                    yield
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
            return _redirect()

        def handle_convert_log_stream(file, url, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
            logs = []
            log_stream = io.StringIO()
            # 下载 PDF
            if url and url.strip():
                pdf_url = url.strip()
                os.makedirs('download', exist_ok=True)
                parsed_url = urlparse(pdf_url)
                filename = os.path.basename(parsed_url.path)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                local_path = os.path.join('download', filename)
                try:
                    logs.append(f"开始下载 PDF: {pdf_url}")
                    yield '\n'.join(logs)
                    r = requests.get(pdf_url)
                    r.raise_for_status()
                    with open(local_path, 'wb') as f2:
                        f2.write(r.content)
                    logs.append(f"PDF 下载完成，已保存到: {os.path.abspath(local_path)}")
                    yield '\n'.join(logs)
                except Exception as e:
                    logs.append(f"下载 PDF 失败: {e}")
                    yield '\n'.join(logs)
                    return
                file_path = local_path
            elif file is not None:
                file_path = file.name if hasattr(file, 'name') else file
            else:
                logs.append('Please upload a file or enter a valid PDF URL.')
                yield '\n'.join(logs)
                return
            # 实时 to_markdown 日志流
            logs.append("开始转换 PDF ...")
            yield '\n'.join(logs)
            import threading
            result_holder = {}
            def run_to_markdown():
                with redirect_stdouterr(log_stream):
                    result_holder['result'] = to_markdown(file_path, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language, logs, log_stream)
            t = threading.Thread(target=run_to_markdown)
            t.start()
            last_log = ''
            while t.is_alive():
                time.sleep(0.5)
                log_stream.flush()
                current_log = log_stream.getvalue()
                # 只在日志有新内容时 yield
                if current_log.strip() and current_log != last_log:
                    logs_with_progress = '\n'.join(logs) + '\n' + current_log
                    yield logs_with_progress
                    last_log = current_log
            t.join()
            # 最终结果
            log_stream.flush()
            final_log = log_stream.getvalue()
            logs_with_progress = '\n'.join(logs)
            if final_log.strip():
                logs_with_progress += '\n' + final_log
            yield logs_with_progress + '\n全部处理完成！'

        def handle_convert_final_outputs(file, url, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
            logs = []
            # 下载 PDF
            if url and url.strip():
                pdf_url = url.strip()
                os.makedirs('download', exist_ok=True)
                parsed_url = urlparse(pdf_url)
                filename = os.path.basename(parsed_url.path)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                local_path = os.path.join('download', filename)
                try:
                    logs.append(f"开始下载 PDF: {pdf_url}")
                    r = requests.get(pdf_url)
                    r.raise_for_status()
                    with open(local_path, 'wb') as f2:
                        f2.write(r.content)
                    logs.append(f"PDF 下载完成，已保存到: {os.path.abspath(local_path)}")
                except Exception as e:
                    logs.append(f"下载 PDF 失败: {e}")
                    return '', '', '', None
                file_path = local_path
            elif file is not None:
                file_path = file.name if hasattr(file, 'name') else file
            else:
                logs.append('Please upload a file or enter a valid PDF URL.')
                return '', '', '', None
            # to_markdown
            md_content, txt_content, archive_zip_path, new_pdf_path, _ = to_markdown(file_path, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language, logs)
            if not (new_pdf_path and os.path.exists(new_pdf_path)):
                return md_content, txt_content, archive_zip_path, None
            return md_content, txt_content, archive_zip_path, new_pdf_path



        change_bu.click(
            fn=handle_convert_log_stream,
            inputs=[file, url_input, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
            outputs=log_box
        )
        change_bu.click(
            fn=handle_convert_final_outputs,
            inputs=[file, url_input, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
            outputs=[md, md_text, output_file, pdf_show]
        )
        clear_bu.add([file, md, pdf_show, md_text, output_file, is_ocr])

    demo.launch(server_name='0.0.0.0')
