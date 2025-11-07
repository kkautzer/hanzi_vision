import { NavLink } from "react-router";

export default function Navbar() {

    function getModelInfo() {
        fetch(`${serverURL}/models/data/${model["model_name"]}`, {
            method: "GET",
        }).then(async (res) => {
            const r = await res.json();
            if (res.status === 200) {
                console.log(r)
                alert("200 OK")
                useNavigate('/training')
                return r
            } else {
                console.log(r)
                alert("Error getting model training information. Please try again later or see the console for more information")
            }
        })
    }

    return <div className="sm:navbar bg-primary text-neutral-content">
        <div className="flex-none">
            <ul className="menu sm:menu-horizontal px-1 primary-case">
                <li><NavLink to='/home'>Home</NavLink></li>
                <li><NavLink to='/evaluate'>Evaluate</NavLink></li>
                <li><NavLink to='/models'>Models</NavLink></li>
                <li><NavLink to='/training'>Training Data</NavLink></li>
                <li><NavLink to='/characters'>View All Characters</NavLink></li>
            </ul>
        </div>
    </div>
}