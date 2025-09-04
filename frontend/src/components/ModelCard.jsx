import { useNavigate } from "react-router";

export default function ModelCard(props) {
    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"

    const model = props.model
    const navigate = useNavigate()
    function getModelInfo() {
        navigate('/training', { state: {preselect_model: model}})
    }


    return <div className="card card-border bg-gray-700 shadow-md text-center" >
        <div className="card-body mx-auto">
            <h2 className="card-title">
                {props.new ? <div className="badge badge-secondary">NEW</div> : ''}
                <strong>{model?.['model_name']}</strong>
            </h2>
            <p>
                <em>{model?.['nchars']} characters</em><br/>
                Highest Validation Accuracy: {model?.['max_val_accuracy']?.toFixed(2)}%<br/>
                Total Epochs: {model?.['epochs']}
            </p>
            <div className="card-actions">
                <button onClick={getModelInfo} className="btn btn-primary btn-block">View Training Data</button>

            </div>
        </div>
    </div>
}