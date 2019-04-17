'use strict';

function CalibrationTool() {

    const requests ={

    };

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
    };

};