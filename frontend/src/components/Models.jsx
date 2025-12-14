import { useState, useEffect } from "react"
import Navbar from "./Navbar"
import ModelCard from "./ModelCard"
export default function Models() {
    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"


    const [ models, setModels ] = useState([])

    const newModels = [
        "InceptionV1-1500-1",
        "InceptionV1-2000-1" 
    ];

    const oldModels = [
        "model-GoogLeNet-500-1.0",
        "model-GoogLeNet-750-1.0",
        "model-500-Inception-v1-SGD",
        "model-750-Inception-v1-SGD"
    ]


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
    
    return <div>
        <div>
            <Navbar />
        </div>
        <div className="mx-[22%] my-4">
            <h1 className="text-4xl text-center">Models & Data</h1>

            <div className="flex flex-col gap-y-2 mt-4">
                {models != oldModels ? models.map((mod) => {
                    if (!oldModels.includes(mod['model_name'])) {
                        return <ModelCard model={mod} new={newModels.includes(mod['model_name'])} key={mod["model_name"]} />
                    } else {
                        return null
                    }
                }) : <p className="text-xl text-red">Loading Models...</p>}
            </div>
        </div>
    </div>
}