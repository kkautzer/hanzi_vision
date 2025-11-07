import { Routes, Route } from 'react-router'
import './index.css'

import Home from './components/Home'
import Evaluation from './components/Evaluation'
import Models from './components/Models'
import TrainingData from './components/TrainingData'
import CharactersList from './components/CharactersList'

function App() {

  // oversee router pages
  return <Routes>
    <Route path='/home?' element={<Home /> } />
    <Route path='/evaluate' element={ <Evaluation />} />
    <Route path='/models' element={<Models />} />
    <Route path='/training' element={<TrainingData />} />
    <Route path='/characters' element={<CharactersList />} />
  </Routes>
}

export default App
