import { Ring } from 'ldrs/react'
import 'ldrs/react/Ring.css'

export default function LoadingAnimationModal(props) {
    const rootStyles = getComputedStyle(document.documentElement)

    return <>
        <dialog id={props.modalId} closedby="none" className="modal shadow-lg bg-transparent bg-opacity-0" onCancel={(e) => e.preventDefault()}>
            <div className="modal-box max-w-5xl bg-transparent aspect-square w-250 flex justify-center items-center">
                {/* modal content goes here */}
                <div className='my-auto mx-auto'>
                    <Ring size={250} speed={1.5} bgOpacity={0} color={rootStyles.getPropertyValue("--color-base-content")}/>
                </div>
            </div>
        </dialog>
    </>
}