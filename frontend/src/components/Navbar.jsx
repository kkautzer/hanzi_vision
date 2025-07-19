import { NavLink } from "react-router";

export default function Navbar() {
    return <div className="sm:navbar bg-primary text-neutral-content">
        <div className="flex-none">
            <ul className="menu sm:menu-horizontal px-1 primary-case">
                <li><NavLink to='/home'>Home</NavLink></li>
                {/* <div className="dropdown dropdown-end">
                    <div tabIndex={0} role='button' className="btn btn-ghost">
                        <div>
                            <p>&darr;Evaluation&darr;</p>
                        </div>
                    </div>
                    <ul tabIndex={0} className="menu menu-sm dropdown-content bg-base-100 text-base-content rounded-box z-1 mt-3 w-52 p-2 shadow"> */}
                        <li><NavLink to='/eval/upload'>Evaluation - Upload</NavLink></li>
                        <li><NavLink to='/eval/drawing'>Evaluation - Drawing</NavLink></li>
                    {/* </ul>
                </div> */}
            </ul>
        </div>
    </div>
}