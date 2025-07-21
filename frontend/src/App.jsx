import { Routes, Route } from 'react-router'
import './index.css'

import Home from './components/Home'
import Evaluation from './components/Evaluation'

function App() {

  // oversee router pages
  return <Routes>
    <Route path='/home?' element={<Home /> } />
    <Route path='/evaluate' element={ <Evaluation />} />
  </Routes>
}

export default App
