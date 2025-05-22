import { BrowserRouter as Router, Link } from "react-router-dom";
import AppRoutes from "./routes"; // Import the routes configuration
import "./App.css";
import QueryProvider from "./context/query-provider";

function App() {
  return (
    <QueryProvider>
      <Router>
        <div className="app-container">
          <nav className="main-nav">
            <ul>
              <li>
                <Link to="/">Home (Existing)</Link> 
              </li>
              <li>
                <Link to="/OpenSourceTools/Extractor/PDF">PDF Extractor (Existing)</Link>
              </li>
              <li>
                <Link to="/pdf-processor">PDF Processor (New)</Link>
              </li>
            </ul>
          </nav>
          <main className="content-area">
            <AppRoutes /> {/* Render the routes here */}
          </main>
        </div>
      </Router>
    </QueryProvider>
  );
}

export default App;
