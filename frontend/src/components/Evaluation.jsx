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
        <div>
            <p>This is the evaluation page -- see below!</p>
            <button onClick={() => setIsDrawing((o) => !o)} className="btn btn-primary">Switch to {(isDrawing) ? "upload" : "drawing"}</button>
        </div>
        <div>
            {(isDrawing) ? <EvalDrawing/> : <EvalUpload />}
        </div>
    </div>

}