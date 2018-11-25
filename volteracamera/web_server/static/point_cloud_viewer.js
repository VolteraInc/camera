/**
 * @author Ryan Wicks
 * Code for a 3D point cloud viewer
 */

"use strict";


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

    var scene, camera, renderer, controls;

    const pointSize = 0.05;
    function init() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000); //Real view, scaled with distance
        renderer = new THREE.WebGLRenderer();

        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        //Resize
        window.addEventListener("resize", function () {
            var width = window.innerWidth;
            var height = window.innerHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        });

        //Adding controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        generatePointCloud();
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
    function renderScene ()  {
      renderer.render( scene, camera );
    };

    var points = [];

    function addPoints ( points_to_add ) {
        points = points.concat();
    };

    function clearPoints() {
        points = [];
    };

    function generatePointCloud() {
        var geometry = new THREE.BufferGeometry();
        
        var numPoints = points.length;

        var positions = new Float32Array( numPoints * 3 );
        var colours = new Float32Array( numPoints * 3 );

        for (var i = 0; i < numPoints; i++) {
            positions[3*i] = point[i].x;
            positions[3*i+1] = point[i].y;
            positions[3*i+2] = point[i].z;

            colours[3*i] = point[i].intensity;
            colours[3*i+1] = 0;
            colours[3*i+2] = 0;
        }

        geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
		geometry.addAttribute( 'color', new THREE.BufferAttribute( colours, 3 ) );
        geometry.computeBoundingBox();
        
        var material = new THREE.PointsMaterial( { size: pointSize, vertexColors: THREE.VertexColors } );
        var pointcloud = new THREE.Points( geometry, material );
    };

    function startPointCapture () {
        const url = "/viewer/points";

        fetch (url, {cache: "no-store"}).then( function(response) {
            if (response.ok) {

                startPointCapture();
            } else {
                console.log("Network request for points failed, response " + response.status + ": " + response.statusText);
                startPointCapture();   
            };

        });
    };    

    function animate() {
        requestAnimationFrame ( animate );

        updateScene();
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
    };

};

//Turn off the image sidebar
var sidebar = document.getElementById('image-sidebar').style.display = "none";

var viewer = PointCloudViewer();
viewer.init()
//Start the viewer communication with the server.
//viewer.startPointCapture();
//start the drawing loop
viewer.animate();

//Create the shape
//      var geometry = new THREE.BoxGeometry(1, 1, 1);
//      var cubeMaterials = [
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Tessa.png"), side: THREE.DoubleSide }), //Right
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Evan.png"), side: THREE.DoubleSide }), //Left
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Marja.png"), side: THREE.DoubleSide }), //Top
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Ryan.png"), side: THREE.DoubleSide }), //Botton
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Tycho.png"), side: THREE.DoubleSide }), //Front
//        new THREE.MeshPhongMaterial({ map: new THREE.TextureLoader().load("img/FamilyDice/Noelle.png"), side: THREE.DoubleSide }) //Back
//      ];     
// 
//      //create material, colour or image texture
//      //var material = new THREE.MeshBasicMaterial( {color: 0xFFFFFF, wireframe: true} );
//      var cube = new THREE.Mesh (geometry, cubeMaterials);
//
//      scene.add( cube );
//
//      camera.position.z = 2;
//
//      //Lighting
//      var ambientLight = new THREE.AmbientLight ( 0xFFFFFF, 0.6 );
//      scene.add (ambientLight);
//      var light1 = new THREE.PointLight (0xFF0044, 0.4, 50);
//      //scene.add (light1);
//      var light2 = new THREE.PointLight (0xFF4400, 0.4, 50);
//      //scene.add (light2);
//      var light3 = new THREE.PointLight (0x44FF00, 0.4, 50);
//      //scene.add (light3);
// 



