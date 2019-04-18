'use strict';

function CalibrationTool() {

    const requests = {
        laser_calibration: "/api/laser_calibration",
        camera_calibration: "/api/camera_calibration",
    };

    let camera_calibration_object;
    let laser_calibration_object;

    (async function startUp () {

        try {
            let laser_response = await fetch(requests.laser_calibration);
            console.log(laser_response);
            let json_object = await laser_response.json();
            handleLaserLoad(json_object);
        } catch (e) {
            console.log (e);
        }

        try {
            let camera_response = await fetch(requests.camera_calibration);
            let json_object = await camera_response.json();
            handleCameraLoad (json_object);
        } catch (e) {
            console.log (e);
        }

    })();


    /**
     * handleCameraLoad load camera object into dialog boxes.
     * @param {*} json_object 
     */
    function handleCameraLoad (json_object) {
        console.log("Camera calibration");
        console.log (json_object);

        if (json_object.__undistort__) {
            document.getElementById("cam_fx").value=json_object.camera_matrix[0][0];
            document.getElementById("cam_fy").value=json_object.camera_matrix[1][1];
            document.getElementById("cam_cx").value=json_object.camera_matrix[0][2];
            document.getElementById("cam_cy").value=json_object.camera_matrix[1][2];
            camera_calibration_object = json_object;
            updateCameraCalibration();
        }
    };

    /**
     * loadCameraCalibration load camera calibration files from file object
     * @param {*} file_object 
     */
    function loadCameraCalibration (file_object) {    
        loadJSONFile(file_object, handleCameraLoad);
    }

    /**
     * updateCameraCalibration update the camera calibration object base on the current camera calibration parameters.
     */
    async function updateCameraCalibration() {
        camera_calibration_object.camera_matrix[0][0] = document.getElementById("cam_fx").value;
        camera_calibration_object.camera_matrix[1][1] = document.getElementById("cam_fy").value;
        camera_calibration_object.camera_matrix[0][2] = document.getElementById("cam_cx").value;
        camera_calibration_object.camera_matrix[1][2] = document.getElementById("cam_cy").value;
        try {
            let json_response = await fetch (requests.camera_calibration, {
                method: 'POST',
                body: JSON.stringify(camera_calibration_object),
                headers: {
                    "Content-Type": "application/json",
                },
            });
        } catch (e) {
            console.log(e);
        };
    };


    /**
     * handleLaserLoad load a laser object into the dialog boxes
     * @param {*} json_object 
     */
    function handleLaserLoad (json_object) {
        console.log("Laser calibration");
        console.log (json_object);

        if (json_object.__plane__) {
            document.getElementById("lp_nx").value = json_object.normal[0];
            document.getElementById("lp_ny").value = json_object.normal[1];
            document.getElementById("lp_nz").value = json_object.normal[2];
            document.getElementById("lp_pz").value = json_object.point[2];
            laser_calibration_object = json_object;
            updateLaserCalibration();
        }
    };

    /**
     * loadLaserCalibration load camera calibration files from file object
     * @param {*} file_object 
     */
    function loadLaserCalibration (file_object) {
        loadJSONFile(file_object, handleLaserLoad);
    }

    /**
     * updateLaserCalibration update the local laser calibration with input values.
     */
    async function updateLaserCalibration() {
        laser_calibration_object.normal[0] = document.getElementById("lp_nx").value;
        laser_calibration_object.normal[1] = document.getElementById("lp_ny").value;
        laser_calibration_object.normal[2] = document.getElementById("lp_nz").value;
        laser_calibration_object.point[2] = document.getElementById("lp_pz").value;
        try {
            let json_response = await fetch (requests.laser_calibration, {
                method: 'POST',
                body: JSON.stringify(laser_calibration_object),
                headers: {
                    "Content-Type": "application/json",
                },
            });
        } catch (e) {
            console.log(e);
        };
    }

    /**
     * loadJSONFile load a json file from the hard drive.
     * @param {*} file_object 
     */
    function loadJSONFile ( file_object, handle_callback ) {
        // Only process json files.
        if (!file_object.type.match('application/json')) {
            console.log(file_object.name + " is not a json file.");
            return;
        }

        var reader = new FileReader();
        // Closure to capture the file information.
        reader.onloadend = function(e) {
            let text = reader.result;
            let json_object = JSON.parse(text);

            handle_callback(json_object);
        };

        // Read in the image file as a data URL.
        reader.readAsText(file_object);
    }

    /**
     * setupTabs setup tab controls by passing in an array of objects that takes a button and tab_element
     * @param {*} tab_control_array {button id:"", tab_element:""}
     */
    function setupTabs( tab_control_array ) {

        let tab_control = tab_control_array;

        for (let i = 0; i < tab_control.length; i++) {
            let button_element = document.getElementById(tab_control[i].button)
            let tab_id = tab_control[i].tab_element;
            button_element.onclick = function () {openTabByID(tab_id)};
        }

        function openTabByID(id) {
            const tabs = Array.from(document.getElementsByClassName('tabs'));
            if (tabs && tabs.length) {
                const matching_tabs = tabs.filter(tab => tab.id === id);
                if (matching_tabs && matching_tabs.length) {
                    tabs.forEach(tab => tab.style.display = "none");
                    matching_tabs[0].style.display = "block";
                }
            }
        }
    };

    /**
     * setFileSelect setup the file select drag in box
     * @param {*} file_select_callback callback function that takes the file and acts on it.
     * @param {*} element_id name of element to drag into.
     */
    function setupFileSelect ( file_select_callback, element_id ) {
    
        let callback_function = file_select_callback;

        let element = document.getElementById (element_id);
        element.addEventListener("drop", handleFileSelect, false);
        element.addEventListener("dragover", handleDragOver, false);


        function handleDragOver(evt) {
            evt.stopPropagation();
            evt.preventDefault();
            evt.dataTransfer.dropEffect = 'copy';
        };

        function handleFileSelect(evt) {
            evt.stopPropagation();
            evt.preventDefault();

            let files = Array.from(evt.dataTransfer.files);
            if (!files.length) { 
                return;
            }   
        
            callback_function (files[0]);
        };

    };

    /**
     * setupMultipleFilesSelect drag in multiple files.
     * @param {*} files_select_callback function that takes a find file reference and acts on it.
     * @param {*} element_id 
     */
    function setupMultipleFilesSelect ( files_select_callback, element_id ) {

        let callback_function = files_select_callback;

        let element = document.getElementById (element_id);
        element.addEventListener("drop", handleMultipleFilesSelect, false);
        element.addEventListener("dragover", handleDragOver, false);

        function handleDragOver(evt) {
            evt.stopPropagation();
            evt.preventDefault();
            evt.dataTransfer.dropEffect = 'copy'; 
        };

        function handleMultipleFilesSelect(evt) {
            evt.stopPropagation();
            evt.preventDefault();
    
            let files = Array.from(evt.dataTransfer.files); 
            if (!files.length) {
                return; //no files passed
            }
            files.forEach( file => callback_function (file) );
        };

    };
    
    return {
        setupTabs: setupTabs,
        setupMultipleFilesSelect: setupMultipleFilesSelect,
        setupFileSelect: setupFileSelect,
        loadCameraCalibration: loadCameraCalibration,
        loadLaserCalibration: loadLaserCalibration,
    };

};