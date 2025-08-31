import Navbar from "./Navbar"

function Home() {
    return <div>
        <div>
            <Navbar />
        </div>
        <div className="mx-[18%] my-4 flex flex-col gap-y-4">
            <div className="text-center">
                <h1 className="text-4xl">Hanzi OCR Project Home Page</h1>
                <div className="mt-2 italic">
                        <p>
                            Project independently led and created by Keagan Kautzer
                        </p>
                        <div className="flex justify-center gap-x-2">
                            <a className='link hover:link-primary' href="https://github.com/kkautzer/">
                                GitHub</a> 
                            <a className='link hover:link-primary' href="https://www.linkedin.com/in/keagan-kautzer/">
                                LinkedIn</a> 
                            <a className='link hover:link-primary' href="mailto:kkautzer05+hanzi_ocr@gmail.com">
                                Email</a>
                        </div>
                    </div>  
            </div>
            <div className="flex gap-x-15">
                <div className="text-center w-[50%] flex flex-col gap-y-4">
                    <h2 className="text-2xl">Project Overview</h2>
                    <p>
                        This project is a solo-developed full-stack application
                        that aims to recognize Chinese characters (Hanzi) using 
                        computer vision and machine learning models. The Underlying
                        model currently utilizes a tweaked GoogLeNet-based 
                        architecture, as this provided the best trade-off between
                        training time and computational resources (my laptop). It 
                        was trained using the CASIA-HWDB1.0 and HWDB1.2 datasets,
                        which (together) contain approximately 2.6 million sample 
                        images and 7,185 distinct Hanzi.
                    </p>
                    <p>
                        This interface also allows users to easily evaluate their own 
                        test images through the <a className='link hover:link-primary' href='/evaluate'>
                        evaluation page</a>. This page has two modes - an upload mode, 
                        where users can upload a photo from their device, and a drawing 
                        mode, where users can draw on their screen, and evaluate the 
                        drawing. 
                    </p>
                    <p className="italic link hover:link-primary"><a href='https://github.com/kkautzer/hanzi_vision'>
                        View Project GitHub Repository
                    </a></p>
                </div>

                <div className="grid gap-y-2">
                    <div>
                        <h2 className="text-2xl">Technologies</h2>
                        <p>This project utilizes:</p>
                        <ul className="list-disc list-inside">
                            <li>PyTorch, for creating, training, and using the model</li>
                            <li>Flask, to link the frontend to the model</li>
                            <li>React, for the frontend interface</li>
                        </ul>
                    </div>
                    <div>
                        <h2 className="text-2xl">Current Goals</h2>
                        <p>Currently, the following goals are in progress: </p>
                        <ul className="list-disc list-inside">
                            <li>Building a mobile interface</li>
                            <li>Incorporating novice-level handwriting into training data</li>
                            <li>Scaling training to 750, 1000, and 1500 characters (currently 500)</li>
                        </ul>
                    </div>
                    <div>
                        <h2 className="text-2xl">Future Directions</h2>
                        <p>There are a few future directions that can be pursued, including: </p>
                        <ul className="list-disc list-inside">
                            <li>Image Segmentation / Multi-Character Recognition</li>
                            <li>Comparative Analysis of Different Underlying Model Architectures</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
}

export default Home