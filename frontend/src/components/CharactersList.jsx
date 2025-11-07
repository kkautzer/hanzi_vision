import { useEffect, useState } from "react";
import Navbar from "./Navbar";

export default function CharactersList(props) {
    // if running locally, use local server, if run on web, use web server
    const serverURL = (window.location.hostname == "localhost")
        ? "http://localhost:5000"
        : "https://hanzi-vision-api.onrender.com"

    const [ chars, setChars ] = useState([]);
    const [ nChars, setNChars ] = useState(100);

    useEffect(() => {
        fetch(`${serverURL}/characters`, {
            method: "GET"
        }).then(async (res) => {
            console.log(res);

            const r = await res.json();
            if (res.status == 200) {
                console.log(r);
                setChars(r);
            } else {
                console.log(r);
                alert("Error in Retrieving Model Data - Please Try Again Later. See Console for More Information")
            }
        })
    }, [])
    
    return <div>
        <Navbar/>

        <div className="mx-15 my-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {chars.slice(0, nChars).map((char) => {
                    return <div key={char?.['frequency_rank']} className="card card-border bg-base-300 border border-base-200 text-base-content shadow-md text-center" >
                        <div className="card-body mx-auto">
                            <h2 className="card-title mx-auto">
                                <strong>{char?.['character']} ({char?.['pinyin']})</strong>
                            </h2>
                            <p>
                                <strong>{char?.['definition']}</strong><br/>
                                <em>#{char?.['frequency_rank']} Most Common Character</em>

                            </p>
                        </div>
                    </div>
                })}
            </div>

            <button onClick={() => setNChars((o) => 2*o) } className="btn btn-primary btn-block mt-4">Show More</button>
        </div>
    </div>
}