import { useRef, useState, useEffect } from 'react'
import { ReactSketchCanvas } from 'react-sketch-canvas'

export default function EvalDrawing() {   
    const canvasRef = useRef(null)
    const [ usePen, setUsePen ] = useState(true)

    function switchTool() {
        if (usePen) {
            canvasRef?.current?.eraseMode(true)
            setUsePen(false)
        } else {
            canvasRef?.current?.eraseMode(false)
            setUsePen(true)
        }
    }

    return <>
        <br/>
        
        <div className='mt-2 space-x-2'>
            <button onClick={() => canvasRef?.current?.clearCanvas()} className='btn btn-error'>Clear</button>
            <button onClick={() => canvasRef?.current?.undo()} className='btn btn-primary'>Undo</button>
            <button onClick={() => canvasRef?.current?.redo()} className='btn btn-primary'>Redo</button>
            <button onClick={switchTool} className='btn btn-primary'>Switch to {(usePen) ? "Eraser" : "Pen"}</button>

        </div>

        <form action={null}>
            <ReactSketchCanvas 
                ref={canvasRef}
                className='mt-2'
                strokeWidth={4}
                strokeColor="white"
                canvasColor='black'
            />

            <br/>
            <button type='submit' className="btn btn-warning mt-2">Evaluate!</button>
        </form>
    </>
}