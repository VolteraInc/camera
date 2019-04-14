/**
 * @author Ryan Wicks
 * Code for a 3D point cloud viewer
 */

"use strict"

//Add control buttons to the top of the viewer
if (WEBGL.isWebGLAvailable() === false) {
    document.body.appendChild(WEBGL.getWebGLErrorMessage());
}


///Performance monitor
(function () {
    var script = document.createElement('script');
    script.onload = function () {
        var stats = new Stats();
        document.body.appendChild(stats.dom);
        requestAnimationFrame(function loop() {
            stats.update();
            requestAnimationFrame(loop)
        });
    };
    script.src = '../static/stats.min.js';
    document.head.appendChild(script);
})();
///end performance monitor

var PointCloudViewer = function () {

    var scene, camera, renderer, controls, hemiLight, dirLight, viewerDiv, axis;

    const pointSize = 0.00000005;
    const pointsName = "points";

    function init(viewerDiv_in) {

        viewerDiv = viewerDiv_in;
        
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, viewerDiv.offsetWidth / viewerDiv.offsetHeight, 0.0000000001, 10); //Real view, scaled with distance
        camera.position.set(0, 0, 0.00005);
        camera.lookAt(scene.position);
        renderer = new THREE.WebGLRenderer();
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(viewerDiv.offsetWidth, viewerDiv.offsetHeight);
        viewerDiv.appendChild(renderer.domElement);
        //document.body.appendChild(renderer.domElement);

        //Resize
        window.addEventListener("resize", function () {
            var width = viewerDiv.offsetWidth;
            var height = viewerDiv.offsetHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        });

        //Adding controls
        controls = new THREE.TrackballControls(camera);
        controls.rotateSpeed = 1.0;
        controls.zoomSpeed = 2;
        controls.panSpeed = 1.8;
        controls.noZoom = false;
        controls.noPan = false;
        controls.staticMoving = true;
        controls.dynamicDampingFactor = 0.3;
        controls.keys = [65, 83, 68];
        controls.addEventListener('change', renderScene);

        //Adding control buttons
        var buttons = [{style:"top:0px; right:0; width:75px;", id:"start", text:"Start", method:startPointCapture},
            {style:"top:25px; right:0; width:75px;", id:"stop", text:"Stop", method:stopPointCapture},
            {style:"top:50px; right:0; width:75px;", id:"save", text:"Save", method:savePoints},
            {style:"top:0px; left:0;", id:"x_pos", text:"x+", method:function () {pointCamera([1, 0, 0])}},
            {style:"top:0px; left:25px;", id:"x_neg", text:"x-", method:function () {pointCamera([-1, 0, 0])}},
            {style:"top:25px; left:0;", id:"y_pos", text:"y+", method:function () {pointCamera([0, 1, 0])}},
            {style:"top:25px; left:25px;", id:"y_neg", text:"y-", method:function () {pointCamera([0, -1, 0])}},
            {style:"top:50px; left:0;", id:"z_pos", text:"z+", method:function () {pointCamera([0, 0, 1])}},
            {style:"top:50px; left:25px", id:"z_neg", text:"z-", method:function () {pointCamera([0, 0, -1])}}];
        
        for (var i = 0; i < buttons.length; i++) {
            var button = document.createElement("button");
            button.style.cssText = buttons[i].style;
            button.id = buttons[i].id;
            button.innerHTML = buttons[i].text;
            button.className = "viewer_class";
            button.addEventListener("click", buttons[i].method, false);
            viewerDiv.appendChild(button);
        }

        scene.background = new THREE.Color().setHSL(0.6, 0, 1);
        scene.fog = new THREE.Fog(scene.background, 1, 5000);

        // LIGHTS
        hemiLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.6);
        hemiLight.color.setHSL(0.6, 1, 0.6);
        hemiLight.groundColor.setHSL(0.095, 1, 0.75);
        hemiLight.position.set(0, 50, 0);
        scene.add(hemiLight);
        dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.color.setHSL(0.1, 1, 0.95);
        dirLight.position.set(- 1, 1.75, 1);
        dirLight.position.multiplyScalar(30);
        scene.add(dirLight);

        axis = new THREE.AxesHelper(0.000005);
        scene.add(axis);

        addPoints([{"x":0, "y":0, "z":0, "i":0}]);
        renderScene();
    };

    /**
     * Point the camera at a origin along a certain direction.
     * @param directionVector direction camera will face
     */
    var pointCamera = function (directionVector) {
        let currentPosition = camera.position;
        let radius = Math.sqrt ( Object.values(currentPosition).reduce( (prev, current)=> prev + current*current, 0 ));
        let new_cam_position = directionVector.map( x => radius*x );
        camera.position.set (...new_cam_position);
        camera.up = new THREE.Vector3(0.0, 1.0, 0.0);
        camera.lookAt(new THREE.Vector3(0, 0, 0));
    };

    var updateScene = function () {
        //cube.rotation.x += 0.01;
        //cube.rotation.y += 0.005;

        //var time = Date.now() + 0.0005;

        //light1.position.x = Math.sin(time*0.7)*30;
        //light1.position.y = Math.cos(time*0.5)*40;
        //light1.position.z = Math.cos(time*0.3)*30;

    };

    //draw scene
    function renderScene() {
        renderer.render(scene, camera);
    };

    var points = []; 
    var center = [0, 0, 0];

    function findCenter(points_to_add) {
        center = [0, 0, 0];

        if ( points_to_add.length === 0 ){
            return;
        }
        for (let point of points_to_add) {
            center[0] += point.x;
            center[1] += point.y;
            center[2] += point.z;
        }
        center[0] /= points_to_add.length;
        center[1] /= points_to_add.length;
        center[2] /= points_to_add.length;
    };

    function addPoints(points_to_add) {
        if (points_to_add !== null) {
            if (center.length === 3 && center.every(function(value, index) { return value === [0, 0, 0][index];})) {
                findCenter(points_to_add);
            }
            points = points.concat(points_to_add);
            generatePointCloud();
        }
    };

    function clearPoints() {
        points = [];
        center = [0, 0, 0];
    };

    function generatePointCloud() {

        //Add new points, so remove old ones first.
        var pointsObject = scene.getObjectByName(pointsName);
        scene.remove(pointsObject);

        var geometry = new THREE.BufferGeometry();

        var numPoints = points.length;

        var positions = new Float32Array(numPoints * 3);
        var colours = new Float32Array(numPoints * 3);

        for (var i = 0; i < numPoints; i++) {
            positions[3 * i] = points[i].x - center[0];
            positions[3 * i + 1] = points[i].y - center[1];
            positions[3 * i + 2] = points[i].z - center[2];

            colours[3 * i] = points[i].i;
            colours[3 * i + 1] = 0;
            colours[3 * i + 2] = 0;
        }

        if (numPoints > 0) {
            geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.addAttribute('color', new THREE.BufferAttribute(colours, 3));
            geometry.computeBoundingBox();

            var material = new THREE.PointsMaterial({ size: pointSize, vertexColors: THREE.VertexColors });
            var pointCloud = new THREE.Points(geometry, material);
            pointCloud.name = pointsName;

            scene.add(pointCloud);
        }
    };

    var capture_points = false;
    function stopPointCapture() {
        const stop_url = "/viewer/stop";
        fetch(stop_url).then(function () {
            console.log("Stopped Point Capture");
            capture_points = false;
        });
    }

    function startPointCapture() {
        const start_url = "/viewer/start";
        if (!capture_points) {
            clearPoints();
            capture_points = true;
            fetch(start_url).then( function () {
                console.log("Starting Point Capture");
                getPoints();
            });
        }
    }

    function savePoints() {
        const concat_points = (accumulator, currentValue) => accumulator + String(currentValue.x) + ", " + 
                                                             String(currentValue.y)+ ", " +String(currentValue.z)+ 
                                                             ", " +String(currentValue.i) + "\n";
        let csv_file = new Blob ([points.reduce (concat_points, "")], {type:"text/csv"});
        let file_url = URL.createObjectURL(csv_file);
        let temp_a = document.createElement("a");
        temp_a.href = file_url;
        temp_a.download = "point_file.csv";
        temp_a.type = "text/csv";
        document.body.appendChild(temp_a);
        temp_a.click(); 
        setTimeout(function() {
            document.body.removeChild(temp_a);
            window.URL.revokeObjectURL(file_url);  
        }, 0);
        
    };

    function getPoints() {
        const url = "/viewer/points";

        if (!capture_points) {
            return;
        }
        fetch(url, { cache: "no-store" }).then(function (response) {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error("Network request for points failed, response " + response.status + ": " + response.statusText);
            };
        }).then(function (myJson) {
            addPoints(myJson);
            setTimeout(getPoints, 250);
        }).catch(function (error) {
            console.log("There was a problem with the request: " + error.message);
        });
    };

    function animate() {
        requestAnimationFrame(animate);

        updateScene();

        controls.update();
        renderScene();
    };

    return {
        //viewer logic
        init: init,
        animate: animate,

        //Point cloud adding logic
        addPoints: addPoints,
        clearPoints: clearPoints,

        //Point getting logic
        startPointCapture: startPointCapture,
        stopPointCapture: stopPointCapture,
    };

};

var viewer = PointCloudViewer();
var viewer_div = document.getElementById("viewer_container");
viewer.init(viewer_div);
//viewer.init();
//viewer.startPointCapture();
//start the drawing loop
viewer.animate();
