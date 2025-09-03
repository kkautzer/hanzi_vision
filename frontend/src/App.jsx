import { Routes, Route } from 'react-router'
import './index.css'

import Home from './components/Home'
import Evaluation from './components/Evaluation'
import Models from './components/Models'
import TrainingData from './components/TrainingData'

function App() {

  // oversee router pages
  return <Routes>
    <Route path='/home?' element={<Home /> } />
    <Route path='/evaluate' element={ <Evaluation />} />
    <Route path='/models' element={<Models />} />
    <Route path='/training' element={<TrainingData />} />
  </Routes>
}

export default App
