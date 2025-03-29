import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./Componentes/Home";
import Predict from "./Componentes/Predict";
import Retrain from "./Componentes/Retrain";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/retrain" element={<Retrain />} />
      </Routes>
    </Router>
  );
}

export default App;
