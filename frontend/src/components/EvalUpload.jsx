
export default function EvalUpload() {
    return <>
        <h1 className="mt-2">Evaluation - Photo Upload Page</h1>
        <form action={null}>
            <input name='image' type='file' className="file-input file-input-primary mt-2" accept='image/*'/>
            <br/>
            <button type='submit' className="btn btn-warning mt-2">Evaluate!</button>
        </form>
    </>
}