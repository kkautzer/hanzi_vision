import EvalDrawing from "./EvalDrawing"
import EvalUpload from "./EvalUpload"
import Navbar from "./Navbar"
import { useState } from "react"

export default function Evaluation() {
    const [ isDrawing, setIsDrawing ] = useState(0)
    
    return <div>
        <div>
            <Navbar/>
        </div>
        <div className="mx-10 my-4 text-center">
            <p className="text-lg">This is the evaluation page - see below!</p>
            <button onClick={() => setIsDrawing((o) => !o)} className="btn btn-primary mt-2">Switch to {(isDrawing) ? "upload" : "drawing canvas"}</button>
            {(isDrawing) ? <EvalDrawing/> : <EvalUpload />}
        </div>
    </div>

}