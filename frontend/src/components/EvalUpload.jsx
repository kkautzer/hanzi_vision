import { useState } from 'react'

export default function EvalUpload(props) {

    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"

    const [ allowSubmit, setAllowSubmit ] = useState(true);

    async function submit(e) {
        e.preventDefault();
        setAllowSubmit(false)

        const formData = new FormData();
        formData.append('image', e.target.image.files[0]);

        props.evaluate(formData)    

        setAllowSubmit(true) 
    }

    return <>

        <h1 className="mt-2">Evaluation - Photo Upload Page</h1>
        <form onSubmit={submit}>
            <input name='image' type='file' className="file-input file-input-primary mt-2" accept='image/*'/>
            <br/>
            <button disabled={!allowSubmit} type='submit' className="btn btn-warning mt-2">Evaluate!</button>
        </form>
    </>
}