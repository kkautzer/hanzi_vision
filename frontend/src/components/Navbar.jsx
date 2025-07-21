import { NavLink } from "react-router";

export default function Navbar() {
    return <div className="sm:navbar bg-primary text-neutral-content">
        <div className="flex-none">
            <ul className="menu sm:menu-horizontal px-1 primary-case">
                <li><NavLink to='/home'>Home</NavLink></li>
                <li><NavLink to='/evaluate'>Evaluate</NavLink></li>
            </ul>
        </div>
    </div>
}