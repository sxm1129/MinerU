import { Routes, Route } from "react-router-dom";
import PDFUpload from "@/pages/extract/components/pdf-upload";
import PDFExtractionJob from "@/pages/extract/components/pdf-extraction";
import PdfProcessorPage from "@/pages/PdfProcessor/PdfProcessorPage"; // Import the new page

function AppRoutes() {
  return (
    <Routes> {/* Changed from <> to <Routes> for proper Route handling */}
      <Route path="/OpenSourceTools/Extractor/PDF" element={<PDFUpload />} />
      <Route
        path="/OpenSourceTools/Extractor/PDF/:jobID"
        element={<PDFExtractionJob />}
      />
      <Route path="/pdf-processor" element={<PdfProcessorPage />} /> {/* Add new route */}
    </Routes>
  );
}

export default AppRoutes;
