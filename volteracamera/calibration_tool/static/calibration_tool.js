'use strict';

function CalibrationTool() {

    const requests = {
        laser_calibration: "/api/laser_calibration",
        camera_calibration: "/api/camera_calibration",
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
            url: "",
            img: new Image,
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

        };

        async function handleFindCalPoints() {

        };

        function handleDestroyItem() {
            camera_calibration_image.loaded = false;
            camera_calibration_image.position_valid = false;
            camera_calibration_image.points_valid = false;
            URL.revokeObjectURL(laser_calibration_image.url);

            let image_item = document.getElementById(camera_calibration_image.name);
            camera_images_tab.removeChild(image_item);
            
            delete camera_calibration_images[camera_calibration_image.name];
        }

        function handleLoadCameraCalibrationImage(url) {
            if (url) {
                camera_calibration_image.url = url;
                camera_calibration_image.loaded = true;
                camera_calibration_image.img.src = url;
            }
            let image_item = document.createElement("li");
            image_item.id = camera_calibration_image.name;
            image_item.className = "list_item";
            let checkbox_item = document.createElement("input");
            checkbox_item.type = "checkbox";
            checkbox_item.className = "inline_checkbox";
            image_item.appendChild(checkbox_item);
            let label_item = document.createElement("label");
            label_item.innerHTML = camera_calibration_image.name;
            image_item.appendChild(label_item);
            camera_images_tab.appendChild(image_item);
            image_item.addEventListener("click", selectCameraImage, false);
            camera_calibration_images[camera_calibration_image.name] = camera_calibration_image;
        };

        if (!(camera_calibration_image.name in camera_calibration_images)) {
           loadImageFile(file_object, handleLoadCameraCalibrationImage);
        }
    };

    function selectCameraImage (e) {
        let key = this.childNodes[1].innerHTML; //second element of list item is the text.
        let item = camera_calibration_images[key];
        if (!item) {
            console.log ("No element with key "+key+" in camera images");
            return;
        }
        if (item.loaded) {
            camera_preview_controls.clear();
            camera_preview_controls.drawImage(item.img);
        }
        if (item.position_valid) {
            camera_preview_controls.drawHeader("(" + item.position[0] + ", " + item.position[1] + ", " + item.position[2] + ")");
        }
        if (item.point_valid) {
            camera_preview_controls.drawPoint(item.image_point, [item.width, item.height]);
        }
    };

    function loadLaserCalibrationImage(file_object) {

        let laser_images_tab = document.getElementById('laser_images_list');

        let laser_calibration_image = {
            name: file_object.name,
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

        };

        async function handleFindLaserPoints() {

        };

        function handleDestroyItem() {
            laser_calibration_image.loaded = false;
            laser_calibration_image.position_valid = false;
            laser_calibration_image.points_valid = false;
            URL.revokeObjectURL(laser_calibration_image.url);
            delete laser_calibration_image.img;

            let image_item = document.getElementById(laser_calibration_image.name);
            laser_images_tab.removeChild(image_item);
            
            delete laser_calibration_images[laser_calibration_image.name];
        }

        function handleLoadLaserCalibrationImage(url) {
            if (url) {
                laser_calibration_image.url = url;
                laser_calibration_image.loaded = true;
                laser_calibration_image.img.src = url;
            }
            let image_item = document.createElement("li");
            image_item.id = laser_calibration_image.name;
            image_item.className = "list_item";
            let checkbox_item = document.createElement("input");
            checkbox_item.type = "checkbox";
            checkbox_item.className = "inline_checkbox";
            image_item.appendChild(checkbox_item);
            let label_item = document.createElement("label");
            label_item.innerHTML = laser_calibration_image.name;
            image_item.appendChild(label_item);
            image_item.addEventListener("click", selectLaserImage, false);
            laser_images_tab.appendChild(image_item);

            laser_calibration_images[laser_calibration_image.name] = laser_calibration_image;
        };

        if (!(laser_calibration_image.name in laser_calibration_images)) {
           loadImageFile(file_object, handleLoadLaserCalibrationImage);
        }
    };

    function selectLaserImage (e) {
        let key = this.childNodes[1].innerHTML; //second element of list item is the text.
        let item = laser_calibration_images[key];
        if (!item) {
            console.log ("No element with key "+key+" in laser images");
            return;
        }
        if (item.loaded) {
            laser_preview_controls.clear();
            laser_preview_controls.drawImage(item.img);
        }
        if (item.position_valid) {
            laser_preview_controls.drawHeader("(" + item.position[0] + ", " + item.position[1] + ", " + item.position[2] + ")");
        }
        if (item.points_valid) {
            laser_preview_controls.drawPoints(item.laser_points, [item.width, item.height]);
        }
    }

    (async function startUp() {

        try {
            let laser_response = await fetch(requests.laser_calibration);
            console.log(laser_response);
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

        var reader = new FileReader();
        // Closure to capture the file information.
        reader.onloadend = function (e) {
            let url = reader.result;

            handle_callback(url);
        };

        // Read in the image file as a data URL.
        reader.readAsDataURL(file_object);
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
            files.forEach(file => callback_function(file));
        };

    };
 
    function setupCameraPreview(canvas_id) {
        camera_preview_controls = setupDisplayCanvas(canvas_id);

        let canvas = document.getElementById(canvas_id);
        let ctx = canvas.getContext("2d");

        camera_preview_controls.drawPoint = function (image_point, original_size) {
            let x = image_point[0] / original_size[0] * canvas.width;
            let y = image_point[1] / original_size[1] * canvas.height;
            ctx.beginPath();
            ctx.fillStyle = "green";
            ctx.arc(x, y, 10, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.fill();
        };
    };

    function setupLaserPreview(canvas_id) {
        laser_preview_controls = setupDisplayCanvas(canvas_id);

        let canvas = document.getElementById(canvas_id);
        let ctx = canvas.getContext("2d");

        laser_preview_controls.drawPoints = function (image_points, original_size) {
            image_points.forEach(image_point => {
                let x = image_point[0] / original_size[0] * canvas.width;
                let y = image_point[1] / original_size[1] * canvas.height;
                ctx.beginPath();
                ctx.fillStyle = "green";
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.fill();
            });
        };
    }

    function setupDisplayCanvas (canvas_id) {

        let canvas = document.getElementById(canvas_id);
        let ctx = canvas.getContext("2d");

        resize();
        canvas.addEventListener("resize", resize);

        function resize(){
            let container = canvas.parentElement;
            //let aspect = canvas.height/canvas.width;
            let width = container.offsetWidth;
            let height = container.offsetHeight;

            canvas.width = width;
            //canvas.height = Math.round(width * aspect);
            canvas.height = height;
        };

        function clear() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        function drawImage (img) {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }

        function drawHeader (header_text) {
            ctx.font = "20px Georgia";
            ctx.fillText("header_text", 10, 50);
        }

        return {
            clear: clear,
            resize: resize,
            drawImage: drawImage,
            drawHeader: drawHeader,
        };
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
    };

};