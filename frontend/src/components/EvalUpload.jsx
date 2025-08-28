import { useState } from 'react'
export default function EvalUpload() {

    const localServerURL = "http://localhost:5000"
    const globalServerURL = 'https://hanzi-vision-api.onrender.com'

    const [ allowSubmit, setAllowSubmit ] = useState(true);

    function submit(e) {
        e.preventDefault();
        setAllowSubmit(false)
        
        const formData = new FormData();
        formData.append('image', e.target.image.files[0]);
        
        fetch(`${globalServerURL}/evaluate`, {
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
            setAllowSubmit(true)
        })
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