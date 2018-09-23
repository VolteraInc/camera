"use strict"

function openNav() {
    document.getElementById("split-content").style.gridTemplateColumns = "20% 80%";
    document.getElementById("toggle-sidebar").onclick=closeNav;
    document.getElementById("toggle-sidebar").innerHTML="\<";
    document.getElementById("hideable-images").style.display = "block";
}

function closeNav() {
   document.getElementById("split-content").style.gridTemplateColumns = "4% 96%";
   document.getElementById("toggle-sidebar").onclick=openNav;
   document.getElementById("toggle-sidebar").innerHTML="\>";
   document.getElementById("hideable-images").style.display = "none";
} 
