import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import NewEvaluationPage from './pages/NewEvaluationPage';
import EvaluationHistoryPage from './pages/EvaluationHistoryPage';

/**
 * Root application component.
 * Sets up React Router v6 with BrowserRouter and defines all routes.
 */
function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          {/* Redirect root to /evaluate */}
          <Route path="/" element={<Navigate to="/evaluate" replace />} />

          {/* New Evaluation form */}
          <Route path="/evaluate" element={<NewEvaluationPage />} />

          {/* History list (no item selected) */}
          <Route path="/history" element={<EvaluationHistoryPage />} />

          {/* History with detail panel open for a specific submission */}
          <Route path="/history/:id" element={<EvaluationHistoryPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
