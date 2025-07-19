import { Routes, Route } from 'react-router'
import './index.css'

import Home from './components/Home'
import EvalUpload from './components/EvalUpload'
import EvalDrawing from './components/EvalDrawing'

function App() {

  // oversee router pages
  return <Routes>
    <Route path='/home?' element={<Home /> } />
    <Route path='/eval/upload' element={ <EvalUpload />} />
    <Route path='/eval/drawing' element={<EvalDrawing />} />
  </Routes>
}

export default App
