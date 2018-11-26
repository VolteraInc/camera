/**
 * @author Ryan Wicks
 * Code for a 3D point cloud viewer
 */

//"use strict";

//Turn off the image sidebar
window.onload = function () {
    var sidebar = document.getElementById('image-sidebar');
    if (sidebar !== null) {
        sidebar.parentNode.removeChild(sidebar);
    }
    var sidebar_script = document.getElementById('image-sidebar-script');
    if (sidebar_script !== null) {
        sidebar_script.parentNode.removeChild(sidebar_script);
    }
};

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

    function init(viewerDiv = null) {

        if (viewerDiv === null) {
            viewerDiv = window;
        }

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, viewerDiv.innerWidth / viewerDiv.innerHeight, 0.0000000001, 10); //Real view, scaled with distance
        camera.position.set(0, 0, 0.00005);
        camera.lookAt(scene.position);
        renderer = new THREE.WebGLRenderer();
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(viewerDiv.innerWidth, viewerDiv.innerHeight);
        document.body.appendChild(renderer.domElement);

        //Resize
        window.addEventListener("resize", function () {
            var width = viewerDiv.innerWidth;
            var height = viewerDiv.innerHeight;
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
//viewer.init("split-content");
viewer.init();
//viewer.startPointCapture();
//start the drawing loop
viewer.animate();
