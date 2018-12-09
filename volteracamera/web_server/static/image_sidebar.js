"use strict"

var Sidebar = Sidebar || {};

//window.closeNav = Sidebar.closeNav;
//window.loadSidebar = Sidebar.loadSidebar;

Sidebar = (function () {
  
  var is_expanded = true;

  function clickNav() {
      if (is_expanded) {
        closeNav();
      } else {
        openNav();
      }
  };

  function openNav() {
    is_expanded = true;
    document.getElementById("split-content").style.gridTemplateColumns = "20% 80%";
    document.getElementById("toggle-sidebar").innerHTML = "\<";
    document.getElementById("hideable-images").style.display = "block";
    document.getElementById("image-sidebar-contents").style.display = "block";
  }

  function closeNav() {
    is_expanded = false;
    document.getElementById("split-content").style.gridTemplateColumns = "4% 96%";
    document.getElementById("toggle-sidebar").innerHTML = "\>";
    document.getElementById("hideable-images").style.display = "none";
    document.getElementById("image-sidebar-contents").style.display = "none";
  }


  function loadSidebar() {
    var url_sidebar = "/load_sidebar";
    var sidebar = document.getElementById('image-sidebar-contents');
    if (sidebar === null) {
      return;
    }
    //Clear out the existing elements.
    while (sidebar.firstChild) {
      sidebar.removeChild(sidebar.firstChild);
    }

    //fetch the latest contents.
    fetch(url_sidebar).then(
      function (response) {
        response.json().then(
          function (data) {
            for (let i = 0; i < data["images"].length; ++i) {
              var divElement = document.createElement("div");
              divElement.setAttribute("id", "sidebar_image");
              var removeImage = document.createElement("a");
              removeImage.setAttribute("href", "#");
              removeImage.onclick = function () { deleteImage(i) };
              removeImage.innerHTML = "&#10005";
              var aElement = document.createElement("a");
              aElement.setAttribute("href", data["images"][i].image);
              var imageElement = document.createElement("img")
              imageElement.setAttribute("src", data["images"][i].thumbnail);
              imageElement.setAttribute("style", "width: 100%");

              divElement.appendChild(removeImage);
              aElement.appendChild(imageElement);
              divElement.appendChild(aElement);
              sidebar.appendChild(divElement);

            }

            if (data["images"].length > 0) {
              var saveElement = document.createElement("a");
              saveElement.innerHTML = "Save Images";
              saveElement.href = "/save_data.zip";
              sidebar.appendChild(saveElement);
            }
          });
      }).catch(
        function (error) {
          console.log("Request failed", error);
        });

  }

  function deleteImage(num) {
    var url = "/delete_image/" + num;
    fetch(url); //don't need to do anything with the response.
    loadSidebar();
  };

  return {
    loadSidebar: loadSidebar,
    deleteImage: deleteImage,
    openNav: openNav,
    closeNav: closeNav,
    clickNav: clickNav
  };

}) ();

