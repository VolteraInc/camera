'use strict';

function CalibrationTool() {

    const requests = {
        laser_calibration: "/api/laser_calibration",
        camera_calibration: "/api/camera_calibration",
        camera_find_point: "/api/camera_find_point",
        parse_camera_position: "/api/parse_camera_position",
        laser_find_points: "/api/find_laser_points",
        parse_laser_height: "/api/parse_laser_height",
        fit_laser: "/api/fit_laser",
        fit_camera: "/api/fit_camera",
        shutdown: "/shutdown",
    };

    let camera_calibration_object;
    let laser_calibration_object;

    let camera_calibration_images = {};
    let laser_calibration_images = {};

    let camera_preview_controls;
    let laser_preview_controls;

    async function loadCameraCalibrationImage(file_object) {
        let camera_images_tab = document.getElementById('camera_images_list');

        let camera_calibration_image = {
            name: file_object.name,
            id: "camera_" + file_object.name,
            url: "",
            img: new Image,
            width: 0,
            height: 0,
            loaded: false,
            position: [],
            position_valid: false,
            image_point: [],
            point_valid: false,

            parseFilename: handleParseFilenameForPosition,
            findLaserPoints: handleFindCalPoints,
            destroyItem: handleDestroyItem,
        };

        async function handleParseFilenameForPosition() {
            try {
                let response = await fetch(requests.parse_camera_position, {
                    method: 'POST',
                    body: JSON.stringify({ filename: camera_calibration_image.name }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                let json_object = await response.json();
                if (!json_object.success) {
                    console.log(json_object.message);
                    return false;
                }
                camera_calibration_image.position = json_object.position;
                camera_calibration_image.position_valid = true;
                let list_item = document.getElementById(camera_calibration_image.id);
                list_item.childNodes[1].style.color = "orange";
            } catch (e) {
                console.log(e);
                return false;
            }
            return true;
        };

        async function handleFindCalPoints() {
            try {
                let response = await fetch(requests.camera_find_point, {
                    method: 'POST',
                    body: JSON.stringify({ image_data: camera_calibration_image.url }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                let json_object = await response.json();
                if (!json_object.success) {
                    console.log(json_object.message);
                    return false;
                }
                camera_calibration_image.image_point = json_object.point;
                camera_calibration_image.point_valid = true;
                let list_item = document.getElementById(camera_calibration_image.id);
                list_item.childNodes[1].style.color = "green";
            } catch (e) {
                console.log(e);
                return false;
            }
            return true;
        };

        function handleDestroyItem() {
            camera_calibration_image.loaded = false;
            camera_calibration_image.position_valid = false;
            camera_calibration_image.points_valid = false;
            try {
                URL.revokeObjectURL(camera_calibration_image.url);
                URL.revokeObjectURL(camera_calibration_image.img.src);
            } catch (e){

            }

            let image_item = document.getElementById(camera_calibration_image.id);
            camera_images_tab.removeChild(image_item);
        }

        function handleLoadCameraCalibrationImage(url) {
            if (url) {
                camera_calibration_image.url = url;
                camera_calibration_image.loaded = true;
                camera_calibration_image.img.src = url;
                camera_calibration_image.width = camera_calibration_image.img.width.valueOf();
                camera_calibration_image.height = camera_calibration_image.img.height.valueOf();
            }
            let image_item = document.createElement("li");
            image_item.id = camera_calibration_image.id;
            image_item.className = "list_item";
            let checkbox_item = document.createElement("input");
            checkbox_item.type = "checkbox";
            checkbox_item.className = "inline_checkbox";
            image_item.appendChild(checkbox_item);
            let label_item = document.createElement("label");
            label_item.innerHTML = camera_calibration_image.name;
            label_item.style.color = "red";
            image_item.appendChild(label_item);
            camera_images_tab.appendChild(image_item);
            image_item.addEventListener("click", setupImageSelector("camera"), false);
            camera_calibration_images[camera_calibration_image.id] = camera_calibration_image;
        };

        if (!(camera_calibration_image.id in camera_calibration_images)) {
            try {
                await loadImageFile(file_object, handleLoadCameraCalibrationImage);
            } catch (e) {
                console.log(e);
                return;
            }
            if (!await handleParseFilenameForPosition()) {
                return;
            }
            if (!await handleFindCalPoints()) {
                return;
            }
        }
    };

    async function loadLaserCalibrationImage(file_object) {

        let laser_images_tab = document.getElementById('laser_images_list');

        let laser_calibration_image = {
            name: file_object.name,
            id: "laser_" + file_object.name,
            url: "",
            img: new Image,
            loaded: false,
            position: [],
            position_valid: false,
            laser_points: [],
            points_valid: false,

            parseFilename: handleParseFilenameForPosition,
            findLaserPoints: handleFindLaserPoints,
            destroyItem: handleDestroyItem,
        };

        async function handleParseFilenameForPosition() {
            try {
                let response = await fetch(requests.parse_laser_height, {
                    method: 'POST',
                    body: JSON.stringify({ filename: laser_calibration_image.name }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                let json_object = await response.json();
                if (!json_object.success) {
                    console.log(json_object.message);
                    return false;
                }
                laser_calibration_image.position = [0, 0, json_object.height];
                laser_calibration_image.position_valid = true;
                let list_item = document.getElementById(laser_calibration_image.id);
                list_item.childNodes[1].style.color = "orange";
            } catch (e) {
                console.log(e);
                return false;
            }
            return true;
        };

        async function handleFindLaserPoints() {
            try {
                let response = await fetch(requests.laser_find_points, {
                    method: 'POST',
                    body: JSON.stringify({ image_data: laser_calibration_image.url }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                let json_object = await response.json();
                if (!json_object.success) {
                    console.log(json_object.message);
                    return false;
                }
                laser_calibration_image.laser_points = json_object.points;
                laser_calibration_image.points_valid = true;
                let list_item = document.getElementById(laser_calibration_image.id);
                list_item.childNodes[1].style.color = "green";
            } catch (e) {
                console.log(e);
                return false;
            }
            return true;
        };

        function handleDestroyItem() {
            laser_calibration_image.loaded = false;
            laser_calibration_image.position_valid = false;
            laser_calibration_image.points_valid = false;
            try {
                URL.revokeObjectURL(laser_calibration_image.url);
                URL.revokeObjectURL(laser_calibration_image.img.src);
            } catch (e) {

            }

            let image_item = document.getElementById(laser_calibration_image.id);
            laser_images_tab.removeChild(image_item);

            delete laser_calibration_images[laser_calibration_image.id];
        }

        function handleLoadLaserCalibrationImage(url) {
            if (url) {
                laser_calibration_image.url = url;
                laser_calibration_image.loaded = true;
                laser_calibration_image.img.src = url;
                laser_calibration_image.width = laser_calibration_image.img.width.valueOf();
                laser_calibration_image.height = laser_calibration_image.img.height.valueOf();
            }
            let image_item = document.createElement("li");
            image_item.id = laser_calibration_image.id;
            image_item.className = "list_item";
            let checkbox_item = document.createElement("input");
            checkbox_item.type = "checkbox";
            checkbox_item.className = "inline_checkbox";
            image_item.appendChild(checkbox_item);
            let label_item = document.createElement("label");
            label_item.innerHTML = laser_calibration_image.name;
            label_item.style.color = "red";
            image_item.appendChild(label_item);
            image_item.addEventListener("click", setupImageSelector("laser"), false);
            laser_images_tab.appendChild(image_item);

            laser_calibration_images[laser_calibration_image.id] = laser_calibration_image;
        };

        if (!(laser_calibration_image.id in laser_calibration_images)) {
            try {
                await loadImageFile(file_object, handleLoadLaserCalibrationImage);
            } catch (e) {
                console.log(e);
                return;
            }
            if (! await handleParseFilenameForPosition()) {
                return;
            }
            if (! await handleFindLaserPoints()) {
                return;
            }
        }
    };

    /**
     * Get the name of the current tab (camera or laser)
     */
    function getCurrentTab() {
        const tabs = Array.from(document.getElementsByClassName('tabs'));
        let active_tab;
        tabs.forEach(tab => {
            if (tab.style.display === "block") {
                active_tab = tab;
            }
        });
        if (active_tab.id.includes("camera")) {
            return "camera";
        } else if (active_tab.id.includes("laser")) {
            return "laser";
        }
        return;
    }

    /**
     * run through the list of available items in the active tab and run item the passed in function on them.
     * @param {function (item)} item_action_function 
     */
    function iterateList(item_action_function) {
        let current_tab = getCurrentTab();
        let list_item = document.getElementById(current_tab + "_images_list");
        for (let i = 0; i < list_item.childNodes.length; i++) {
            let item = list_item.childNodes[i];
            if (item && item.nodeName === "LI") {
                item_action_function(item);
            }
        }
    };

    /**
     * Returns the function that sets up and populates the 3d viewer.
     * @param {laser or camera} which_type 
     */
    function setupUpdatePreviewView (which_type) {

        async function handleUpdateCameraPreviewViewer() {
            let items = Object.values(camera_calibration_images);
            let points_3d_actual = [];
            let points_2d_observed = [];
            for (const item of items) {
                if (item.point_valid && item.position_valid) {
                    points_3d_actuals.push (item.position);
                    points_2d_observed.push (item.point);
                }
            }
            let response_json;
            try {
                let response = await fetch (requests.preview_camera_solution, {
                    method: "POST",
                    body: JSON.stringify(),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                response_json = await response.json();
            } catch (e) {
                console.log(e);
                return;
            }

            if ( !response_json['success'] ) {
                console.log (response_json['message'])
                return;
            }
        }

        function handleUpdateLaser3DViewer() {
            laser_preview_controls.show3d();
        }

        if (which_type === "camera") {
            return handleUpdateCameraPreviewViewer;
        } else if (which_type === "laser") {
            return handleUpdateLaser3DViewer;
        }
    }

    /**
     * returns the image updater for the camera or laser.
     * @param {string} which_type (camera or laser) 
     */
    function setupUpdateImage(which_type) {

        function handleUpdateLaserImage(item) {
            laser_preview_controls.clear();
            if (item.loaded) {
                laser_preview_controls.drawImage(item.img);
            }
            if (item.position_valid) {
                laser_preview_controls.drawHeader("(" + item.position[0] + ", " + item.position[1] + ", " + item.position[2] + ")");
            }
            if (item.points_valid) {
                laser_preview_controls.drawPoints(item.laser_points);
            }
        };

        function handleUpdateCameraImage(item) {
            camera_preview_controls.clear();
            if (item.loaded) {
                camera_preview_controls.drawImage(item.img);

                //You only get 1. Saves memory.
                try {
                    URL.revokeObjectURL(item.url);
                    URL.revokeObjectURL(item.img);
                } catch (e) {

                }
                item.loaded = false;
            }
            if (item.position_valid) {
                camera_preview_controls.drawHeader("(" + item.position[0] + ", " + item.position[1] + ", " + item.position[2] + ")");
            }
            if (item.point_valid) {
                camera_preview_controls.drawPoint(item.image_point);
            }
        }

        if (which_type === "camera") {
            return handleUpdateCameraImage;
        } else if (which_type === "laser") {
            return handleUpdateLaserImage;
        }
    }


    /**
     * Generates the function for handling image selection using a mouse.
     * @param {string} which_type (camera or laser)
     */
    function setupImageSelector(which_type) {

        let type_string = which_type;
        let handleUpdateImage = setupUpdateImage(which_type);

        function selectImage(e) {
            let key = type_string + "_" + this.childNodes[1].innerHTML; //second element of list item is the text.
            let item;
            if (type_string === "camera") {
                item = camera_calibration_images[key];
            } else {
                item = laser_calibration_images[key];
            }
            if (!item) {
                console.log("No element with key " + key + " in " + type_string + " images");
                return;
            }
            iterateList((item)=>{
                item.classList.remove("selected_item")
            });
            document.getElementById(item.id).classList.add("selected_item");
            handleUpdateImage(item);
        };

        return selectImage;
    };

    /**
     * Handle the up down key presses.
     * @param {event} evt 
     */
    function handleKeyPress (evt) {
        let selected_values = Array.from(document.getElementsByClassName("selected_item"));
        let tab_type = getCurrentTab();
        selected_values = selected_values.filter((item)=>item.id.includes(tab_type));
        //filter by laser or camera
        if (evt.code === "ArrowDown") {
            if (selected_values.length !== 0) { //items are already selected
                let selected_item = selected_values[0];
                let next_item = selected_item.nextSibling;
                if (next_item && next_item.nodeName === "LI") {
                    iterateList((item) => { item.classList.remove("selected_item") })
                    next_item.classList.add("selected_item")
                    let next_image = tab_type === "laser" ? laser_calibration_images[next_item.id] : camera_calibration_images[next_item.id];
                    setupUpdateImage(tab_type)(next_image);
                }
            } else {
                let selected_item = document.getElementById(tab_type + "_images_list").firstChild;
                if (selected_item && selected_item.nodeName === "LI") {
                    selected_item.classList.add("selected_item");
                    let image = tab_type === "laser" ? laser_calibration_images[selected_item.id] : camera_calibration_images[selected_item.id];
                    setupUpdateImage(tab_type)(image);
                }
            }
        }
        else if (evt.code === "ArrowUp") {
            if (selected_values.length !== 0) { //items are already selected
                let selected_item = selected_values[0];
                let previous_item = selected_item.previousSibling;
                if (previous_item && previous_item.nodeName === "LI") {
                    iterateList((item) => { item.classList.remove("selected_item") })
                    previous_item.classList.add("selected_item")
                    let previous_image = tab_type === "laser" ? laser_calibration_images[previous_item.id] : camera_calibration_images[previous_item.id];
                    setupUpdateImage(tab_type)(previous_image);
                }
            } else {
                let selected_item = document.getElementById(tab_type + "_images_list").lastChild;
                if (selected_item && selected_item.nodeName === "LI") {
                    selected_item.classList.add("selected_item");
                    let image = tab_type === "laser" ? laser_calibration_images[selected_item.id] : camera_calibration_images[selected_item.id];
                    setupUpdateImage(tab_type)(image);
                }
            }
}
    }

    /**
     * startup function, called on initialization.
     */
    (async function startUp() {

        try {
            let laser_response = await fetch(requests.laser_calibration);
            let json_object = await laser_response.json();
            handleLaserLoad(json_object);
        } catch (e) {
            console.log(e);
        }

        try {
            let camera_response = await fetch(requests.camera_calibration);
            let json_object = await camera_response.json();
            handleCameraLoad(json_object);
        } catch (e) {
            console.log(e);
        }

        document.addEventListener('keydown', handleKeyPress);
    })();


    /**
     * handleCameraLoad load camera object into dialog boxes.
     * @param {*} json_object 
     */
    function handleCameraLoad(json_object) {
        console.log("Camera calibration");
        console.log(json_object);

        if (json_object.__undistort__) {
            document.getElementById("cam_fx").value = json_object.camera_matrix[0][0];
            document.getElementById("cam_fy").value = json_object.camera_matrix[1][1];
            document.getElementById("cam_cx").value = json_object.camera_matrix[0][2];
            document.getElementById("cam_cy").value = json_object.camera_matrix[1][2];
            camera_calibration_object = json_object;
            updateCameraCalibration();
        }
    };

    /**
     * loadCameraCalibration load camera calibration files from file object
     * @param {*} file_object 
     */
    function loadCameraCalibration(file_object) {
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
            let json_response = await fetch(requests.camera_calibration, {
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
    function handleLaserLoad(json_object) {
        console.log("Laser calibration");
        console.log(json_object);

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
    function loadLaserCalibration(file_object) {
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
            let json_response = await fetch(requests.laser_calibration, {
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
    function loadJSONFile(file_object, handle_callback) {
        // Only process json files.
        if (!file_object.type.match('application/json')) {
            console.log(file_object.name + " is not a json file.");
            return;
        }

        var reader = new FileReader();
        // Closure to capture the file information.
        reader.onloadend = function (e) {
            let text = reader.result;
            let json_object = JSON.parse(text);

            handle_callback(json_object);
        };

        // Read in the json file object.
        reader.readAsText(file_object);
    }

    function loadImageFile(file_object, handle_callback) {
        //only process images
        if (!file_object.type.match('image')) {
            console.log(file_object.name + " is not an image file.");
            return;
        }
        return new Promise((resolve, reject) => {
            var reader = new FileReader();
            // Closure to capture the file information.
            reader.onloadend = function (e) {
                let url = reader.result;
                handle_callback(url);
                resolve();
            };

            // Read in the image file as a data URL.
            reader.readAsDataURL(file_object);
        });
    }

    /**
     * setupTabs setup tab controls by passing in an array of objects that takes a button and tab_element
     * @param {*} tab_control_array {button id:"", tab_element:""}
     */
    function setupTabs(tab_control_array) {

        let tab_control = tab_control_array;

        for (let i = 0; i < tab_control.length; i++) {
            let button_element = document.getElementById(tab_control[i].button)
            let tab_id = tab_control[i].tab_element;
            button_element.onclick = function () { openTabByID(tab_id) };
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
            laser_preview_controls.resize();
            camera_preview_controls.resize();
        }
    };

    /**
     * setFileSelect setup the file select drag in box
     * @param {*} file_select_callback callback function that takes the file and acts on it.
     * @param {*} element_id name of element to drag into.
     */
    function setupFileSelect(file_select_callback, element_id) {

        let callback_function = file_select_callback;

        let element = document.getElementById(element_id);
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

            callback_function(files[0]);
        };

    };

    /**
     * setupMultipleFilesSelect drag in multiple files.
     * @param {*} files_select_callback function that takes a find file reference and acts on it.
     * @param {*} element_id 
     */
    function setupMultipleFilesSelect(files_select_callback, element_id) {

        let callback_function = files_select_callback;

        let element = document.getElementById(element_id);

        if (element.type === "file") {
            element.addEventListener("change", handleMultipleFilesSelect, false);
        } else {
            element.addEventListener("drop", handleMultipleFilesSelect, false);
            element.addEventListener("dragover", handleDragOver, false);
        }

        function handleDragOver(evt) {
            evt.stopPropagation();
            evt.preventDefault();
            evt.dataTransfer.dropEffect = 'copy';
        };

        function handleMultipleFilesSelect(evt) {
            evt.stopPropagation();
            evt.preventDefault();

            let files = []
            try {
                files = Array.from(evt.dataTransfer.files);
            } catch (e) {
                files = Array.from(evt.target.files);
            }
            if (!files.length) {
                return; //no files passed
            }
            files.forEach(file => callback_function(file));
        };

    };

    function setupCameraPreview(canvas_id_2d, canvas_id_3d) {
        camera_preview_controls = setupDisplayCanvas(canvas_id_2d, canvas_id_3d);

        let canvas_2d = document.getElementById(canvas_id_2d);
        let canvas_3d = document.getElementById(canvas_id_3d);
        let ctx = canvas_2d.getContext("2d");

        camera_preview_controls.drawPoint = function (image_point) {
            let x = image_point[0] / camera_preview_controls.image_width * canvas_2d.width;
            let y = image_point[1] / camera_preview_controls.image_height * canvas_2d.height;
            ctx.beginPath();
            ctx.fillStyle = "green";
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.fill();
        };

        camera_preview_controls.drawResidual = function (projected_3d_points, image_points) {
            camera_preview_controls.clear();
            if (projected_3d_points.length !== image_points.length) {
                console.log ("Mismatch in point vectors. Can't draw residuals.");
                return;
            }
            for (let i = 0; i < image_points.length; ++i) {
                let xi = image_points[i][0] / laser_preview_controls.image_width * canvas_2d.width;
                let yi = image_point[i][1] / laser_preview_controls.image_height * canvas_2d.height;
                ctx.beginPath();
                ctx.fillStyle = "green";
                ctx.arc(xi, yi, 2, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.fill();
                ctx.fillStyle = "red";
                let xp = projected_3d_points[i][0] / laser_preview_controls.image_width * canvas_2d.width;
                let yp = projected_3d_points[i][1] / laser_preview_controls.image_height * canvas_2d.height;
                ctx.lineTo(xp, yp);
                ctx.stroke();
                ctx.fillStyle = "green";
                ctx.arc(xp, yp, 2, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.fill();
                ctx.closePath();
            }

        }
    };

    function setupLaserPreview(canvas_id_2d, canvas_id_3d) {
        laser_preview_controls = setupDisplayCanvas(canvas_id_2d, canvas_id_3d);

        let canvas_2d = document.getElementById(canvas_id_2d);
        let canvas_3d = document.getElementById(canvas_id_3d);
        let ctx = canvas_2d.getContext("2d");

        laser_preview_controls.drawPoints = function (image_points) {
            image_points.forEach(image_point => {
                let x = image_point[0] / laser_preview_controls.image_width * canvas_2d.width;
                let y = image_point[1] / laser_preview_controls.image_height * canvas_2d.height;
                ctx.beginPath();
                ctx.fillStyle = "green";
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.fill();
            });
        };
    }

    function setupDisplayCanvas(canvas_2d_id, canvas_3d_id) {

        let canvas = document.getElementById(canvas_2d_id);
        let canvas_3d = document.getElementById(canvas_3d_id);
        let ctx = canvas.getContext("2d");
        let image_width;
        let image_height;

        resize();
        canvas.addEventListener("resize", resize);

        function resize() {
            let container = canvas.parentElement;
            //let aspect = canvas.height/canvas.width;
            let width = container.offsetWidth;
            let height = container.offsetHeight;

            canvas.width = width;
            canvas_3d.width = width;
            //canvas.height = Math.round(width * aspect);
            canvas.height = height;
            canvas_3d.height = height;
        };

        function clear() {
            canvas.style.display = "block";
            canvas_3d.style.display = "none";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        function drawImage(img) {
            canvas.style.display = "block";
            canvas_3d.style.display = "none";
            this.image_width = img.width;
            this.image_height = img.height.valueOf();
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }

        function drawHeader(header_text) {
            canvas.style.display = "block";
            canvas_3d.style.display = "none";
            ctx.font = "20px Georgia";
            ctx.fillStyle = "black";
            ctx.fillText(header_text, 10, 30);
        }

        function show3d() {
            canvas.style.display = "none";
            canvas_3d.style.display = "block";
        }

        return {
            clear: clear,
            resize: resize,
            drawImage: drawImage,
            drawHeader: drawHeader,
            image_width: image_width,
            image_height: image_height,
            show3d: show3d,
        };
    }

    function setupCameraDeleteButton(button_id, list_ul_id) {
        setupDeleteButton(button_id, list_ul_id, "camera");
    }

    function setupLaserDeleteButton(button_id, list_ul_id) {
        setupDeleteButton(button_id, list_ul_id, "laser");
    }

    function setupDeleteButton(button_id, list_ul_id, tab_name) {
        let button = document.getElementById(button_id);

        button.addEventListener("click", function () {
            let list = document.getElementById(list_ul_id);

            if (!list.childNodes || list.childNodes.length == 0) return;

            for (let i = 0; i < list.childNodes.length; i++) {
                let item = list.childNodes[i];
                if (item.nodeName == "LI") {
                    if (item.firstChild.checked) {
                        let key = tab_name + "_" + item.childNodes[1].innerHTML;
                        if (tab_name === "camera") {
                            camera_calibration_images[key].destroyItem();
                        } else {
                            laser_calibration_images[key].destroyItem();
                        }
                        i--;
                    }
                }
            }
        }, false);
    }

    /**
     * Setup the select all button. Same function for both tabs.
     * @param {*} button_id 
     */
    function setupSelectAll(button_id) {
        document.getElementById(button_id).addEventListener("click", () => {
            iterateList((item) => { item.firstChild.checked = true; });
        }, false);
    }

    /**
     * Sets up shut down button that stops server.
     * @param {*} shutdown_button_id 
     */
    function setupShutdownButton (shutdown_button_id){
        document.getElementById(shutdown_button_id).addEventListener("click", ()=>{window.location.href = requests.shutdown});
    }

    /**
     * setup the button that will intitialize the camera fitting.
     * @param {string of the camera fit button id} camera_fit_button_id 
     */
    function setupCameraFitButton (camera_fit_button_id){
        let button = document.getElementById(camera_fit_button_id);

        async function handleCameraFit() {
            let response;
            let response_json;
            button.disabled = true;
            let positions = [];
            let points = [];
            const items = Object.values(camera_calibration_images);
            for (const item of items) {
                if (item.position_valid && item.point_valid) {
                    positions.push(item.position);
                    points.push(item.image_point);
                }
            }
            try {
                response = await fetch(requests.fit_camera, {
                    method: "POST",
                    body: JSON.stringify({
                        positions: positions,
                        points: points,
                        calibration: camera_calibration_object,
                        rvec: rx_tx_controls['camera_rot'].getArray(),
                        tvec: rx_tx_controls['camera_trans'].getArray(),
                    }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                response_json = await response.json();
            } catch (e) {
                console.log(e);
                button.disabled = false;
                return
            }
            if (!response_json.success) {
                console.log (response_json.message);
                button.disabled = false;
                return false;
            }
            button.disabled = false;
            //Temporary output.
            console.log(response_json);
        }

        button.addEventListener("click", handleCameraFit, false);
    }

    /**
     * setup the button that will intitialize the Laser fitting.
     * @param {string of the Laser fit button id} Laser_fit_button_id 
     */
    function setupLaserFitButton (laser_fit_button_id){
        let button = document.getElementById(laser_fit_button_id);

        async function handleLaserFit() {
            button.disabled = true;
            let response;
            let response_json;
            let heights = [];
            let points = [];
            const items = Object.values(laser_calibration_images);
            for (const item of items) {
                if (item.position_valid && item.points_valid) {
                    heights.push(item.heights);
                    points.push(item.points);
                }
            }
            try {
                response = await fetch(requests.fit_laser, {
                    method: "POST",
                    body: JSON.stringify({
                        heights: heights,
                        points: points,
                        camera_calibration: camera_calibration_object,
                        laser_calibration: laser_calibration_object,
                        rvec: rx_tx_controls['laser_rot'].getArray(),
                        tvec: rx_tx_controls['laser_trans'].getArray(),
                    }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });
                response_json = await response.json();
            } catch (e) {
                console.log(e);
                button.disabled = false;
                return
            }
            if (!response_json.success) {
                console.log (response_json.message);
                button.disabled = false;
                return false;
            }
            button.disabled = false;
            //temporary output
            console.log(response_json)
        }

        button.addEventListener("click", handleLaserFit, false);
    }

    /**
     * Set up the handlers for retrieving and setting vectors.
     * @param {x element} x_id 
     * @param {y element} y_id 
     * @param {z element} z_id 
     * @param {laser or camera, tx or rx} system 
     */
    let rx_tx_controls = {}
    function setupVectorControls(x_id, y_id, z_id, system) {
        let x_elem = document.getElementById(x_id);
        let y_elem = document.getElementById(y_id);
        let z_elem = document.getElementById(z_id);

        rx_tx_controls[system] = {
            getX: ()=>{return x_elem.value},
            getY: ()=>{return y_elem.value},
            getZ: ()=>{return z_elem.value},
            setX: (x)=>{x_elem.value = x;},
            setY: (y)=>{y_elem.value = y;},
            setZ: (z)=>{z_elem.value = z;},
            getArray: ()=>{return [x_elem.value, y_elem.value, z_elem.value];},
            setArray: (vec)=>{
                x_elem.value = vec[0];
                y_elem.value = vec[1];
                z_elem.value = vec[2];
            },
        }
    }

    /**
     * setup the 3d preview button for camera images.
     */
    function setup3DPreviewButton(update_button_id, tab_id) {
        let button = document.getElementById(update_button_id);
        let update3DView = setupUpdate3DView(tab_id)
        button.addEventListener("click", ()=>{
            update3DView();
        }, false);
    }


    return {
        setupTabs: setupTabs,
        setupMultipleFilesSelect: setupMultipleFilesSelect,
        setupFileSelect: setupFileSelect,
        loadCameraCalibration: loadCameraCalibration,
        loadLaserCalibration: loadLaserCalibration,
        loadCameraCalibrationImage: loadCameraCalibrationImage,
        loadLaserCalibrationImage: loadLaserCalibrationImage,
        setupCameraPreview: setupCameraPreview,
        setupLaserPreview: setupLaserPreview,
        setupCameraDeleteButton: setupCameraDeleteButton,
        setupLaserDeleteButton: setupLaserDeleteButton,
        setupSelectAll: setupSelectAll,
        setupShutdownButton: setupShutdownButton,
        setupCameraFitButton: setupCameraFitButton,
        setupLaserFitButton: setupLaserFitButton,
        setupVectorControls: setupVectorControls,
        setup3DPreviewButton: setup3DPreviewButton,
    };

};