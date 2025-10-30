import { useLocation } from "react-router";
import Navbar from "./Navbar";
import { useEffect, useState } from "react";
import TrainingDataChart from "./TrainingDataChart";

export default function TrainingData(props) {
    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"

    const location = useLocation()
    const INITIAL_MODEL = location.state?.["preselect_model"]['model_name'] || ""

    const [ models, setModels ] = useState([])
    const [ data, setData ] = useState([])
    const [model, setModel ] = useState(INITIAL_MODEL)
    
    useEffect(() => {
        fetch(`${serverURL}/models`, {
            method: "GET"
        }).then(async (res) => {
            const r = await res.json()
            if (res.status == 200) {
                setModels(r)
                if (!INITIAL_MODEL) {
                    setModel(r[0]['model_name'])
                }
            } else {
                console.log(r)
                alert("Error in Retrieving Model Data - Please Try Again Later. See Console for More Information")
            }
        })
    }, [])

    useEffect(() => {
        if (!model) {
            return
        }

        fetch(`${serverURL}/models/data/${model}`, {
            method: "GET",
        }).then(async (res) => {
            if (res.status === 200) {
                const r = await res.json();
                setData(r)
            } else {
                console.log(res)
                alert("Error getting model training information. Please try again later or see the console for more information")
            }
        })
    }, [model, models])

    return <div>
        <div>
            <Navbar />
        </div>
        <div className="mx-[22%] my-4">
            <h1 className="text-4xl text-center">TRAINING DATA PAGE</h1>

            <div className="mt-4">
                {models.length > 0 ? 
                    <select onChange={(s) => setModel(s.target.value)} value={model} id='evalModel' name='evalModel' className="bg-base-300 border border-base-200 text-base-content text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block mx-auto p-2.5">
                        {models.map((model) => {
                            return <option key={model["model_name"]} value={model['model_name']}>{model['model_name']}</option>
                        })}
                    </select>
                : <p className="text-md mt-2"><em>Loading Models...</em></p> 
                }
            </div>
                

            <div className="mt-4 text-center">
                <TrainingDataChart data={data} />


                {/* REMOVE ( OR REFORMAT ) THE BELOW IN THE FUTURE */}
                {/* <ul>

                    {data.map((entry) => {

                        return <li key={`${entry["epoch"]}+${entry['val_accuracy']}`} ><strong>Epoch {entry['epoch']}</strong>: <em>{entry['val_accuracy'].toFixed(2)}% Validation Accuracy</em></li>
                    })}
                </ul> */}
            </div>
        </div>

    </div>
}