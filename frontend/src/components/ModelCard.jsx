


export default function ModelCard(props) {
    console.log(props.model)
    const model = props.model

    function getModelInfo() {
        
        fetch("", {
            method: "GET"
        }).then(async (res) => {
            const r = await res.json();
            if (res.status === 200) {
                console.log(r)
                /// display model info or set state variable to display model information
                return r
            } else {
                console.log(r)
                alert("Error getting model training information. Please try again later or see the console for more information")
            }
        })
        return
    }


    return <div className="card card-border bg-gray-700 shadow-md text-center" >
        <div className="card-body mx-auto">
            <h2 className="card-title">
                {props.new ? <div className="badge badge-secondary">NEW</div> : ''}
                <strong>{model?.['model_name']}</strong>
            </h2>
            <p>
                <em>{model?.['nchars']} characters</em><br/>
                Highest Validation Accuracy: {model?.['max_val_accuracy']}%<br/>
                Total Epochs: {model?.['epochs']}
            </p>
            <div className="card-actions">
                <button onClick={/*getModelInfo*/null} className="btn btn-primary btn-block">View Training Graph</button>

            </div>
        </div>
    </div>
}