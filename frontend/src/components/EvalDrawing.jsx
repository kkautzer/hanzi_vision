import { useRef, useState, useEffect } from 'react'
import { ReactSketchCanvas } from 'react-sketch-canvas'
import LoadingAnimationModal from './LoadingAnimationModal'

export default function EvalDrawing() {
    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"

    const canvasRef = useRef(null)
    const [ usePen, setUsePen ] = useState(true)
    const [ allowSubmit, setAllowSubmit ] = useState(true)

    function switchTool() {

        if (usePen) {
            canvasRef?.current?.eraseMode(true)
            setUsePen(false)
        } else {
            canvasRef?.current?.eraseMode(false)
            setUsePen(true)
        }
    }

    function dataURItoBlob(dataURI) {
            const byteString = atob(dataURI.split(',')[1]);
            const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]; // gets encoding (e.g. 'base64')
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            return new Blob([ab], { type: mimeString });
        }

    async function submit(e) {
        e.preventDefault();

        document.getElementById('drawingLoadingModal').showModal();
        setAllowSubmit(false);

        const imgURI = await canvasRef.current.exportImage("png");

        const blob = dataURItoBlob(imgURI);

        const formData = new FormData();
        formData.append('image', blob);
        fetch(`${serverURL}/evaluate`, {
            method: "POST",
            body: formData
        }).then(async (res) => {
            console.log(res)
            const r = await res.json();
            if (res.status === 200) {
                console.log(r)
                alert(`Predicted character: ${r.label}`)
            } else {
                alert(r.message)
            }
        }).then(() => {
            setAllowSubmit(true);
            document.getElementById('drawingLoadingModal').close();
        }).catch(err => {
            alert(err);
            setAllowSubmit(true)
            document.getElementById('drawingLoadingModal').close();

        })
    }

    return <>
        <LoadingAnimationModal modalId={"drawingLoadingModal"} />
        
        <h1 className="mt-2">Evaluation - Photo Drawing Page</h1>
        <div className='mt-2 space-x-2'>
            <button onClick={() => canvasRef?.current?.clearCanvas()} className='btn btn-error'>Clear</button>
            <button onClick={() => canvasRef?.current?.undo()} className='btn btn-primary'>Undo</button>
            <button onClick={() => canvasRef?.current?.redo()} className='btn btn-primary'>Redo</button>
            <button onClick={switchTool} className='btn btn-primary'>Switch to {(usePen) ? "Eraser" : "Pen"}</button>

        </div>

        <form onSubmit={submit}>
            <ReactSketchCanvas style={{aspectRatio:"1/1"}} 
                ref={canvasRef}
                className='mt-4 mx-auto'
                strokeWidth={4}
                strokeColor="black"
                // canvasColor='black'
                width="300px"
            />

            <br/>
            <button disabled={!allowSubmit} type='submit' className="btn btn-warning mt-2">Evaluate!</button>
        </form>
    </>
}