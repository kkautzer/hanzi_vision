import EvalDrawing from "./EvalDrawing"
import EvalUpload from "./EvalUpload"
import Navbar from "./Navbar"
import { useState, useEffect, useRef } from "react"
import LoadingAnimationModal from './LoadingAnimationModal'

export default function Evaluation() {

    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"


    const [ isDrawing, setIsDrawing ] = useState(0)
    const [ models, setModels ] = useState([])
    const modelRef = useRef(null)
    const DEFAULT_MODEL_NAME = "model-GoogLeNet-750-2.0"

    useEffect(() => {
        fetch(`${serverURL}/models`, {
            method: "GET"
        }).then(async (res) => {
            const r = await res.json()
            if (res.status == 200) {
                console.log(r)
                setModels(r)
            } else {
                console.log(r)
                alert("Error in Retrieving Model Data - Please Try Again Later. See Console for More Information")
            }
        })
    }, [])
    
    function run_eval(formData) {
        document.getElementById("loadingModal").showModal();

        // edit form data to add model name
        formData.append('model', modelRef.current.value)

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
            // setAllowSubmit(true);
            document.getElementById('loadingModal').close();
        }).catch(err => {
            alert(err);
            // setAllowSubmit(true)
            document.getElementById('loadingModal').close();
            return
        })
    }

    return <div>
        <LoadingAnimationModal modalId={"loadingModal"} />

        <div>
            <Navbar/>
        </div>
        <div className="mx-10 my-4 text-center">
            <h4 className="text-xl">Choose Evaluation Model &darr;</h4>

            {models.length > 0 ? 
                <select ref={modelRef} id='evalModel' name='evalModel' defaultValue={DEFAULT_MODEL_NAME} className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block mx-auto p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500">
                    {models.map((model) => {
                        return <option key={model["model_name"]} value={model['model_name']}>{model['model_name']}</option>
                    })}
                </select>
            : <p className="text-md mt-2"><em>Loading Models...</em></p> 
            }
            <button onClick={() => setIsDrawing((o) => !o)} className="btn btn-primary mt-4 ">Switch to {(isDrawing) ? "upload" : "drawing canvas"}</button>
            {(isDrawing) ? <EvalDrawing evaluate={run_eval} /> : <EvalUpload evaluate={run_eval} />}
        </div>
    </div>

}