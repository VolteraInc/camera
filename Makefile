IP_ADDRESS = ${VOLTERA_CAMERA_PI_IP}

run:
	FLASK_APP=volteracamera.web_server \
	FLASK_DEBUG=1 \
	flask run --host=0.0.0.0

run_calibration:
	FLASK_APP=volteracamera.calibration_tool \
	FLASK_DEBUG=1 \
	flask run

update-requirements:
	pip freeze > Requirements.txt

install-requirements:
	pip install -r Requirements.txt

setup-venv:
	python3 -m venv venv

copy-to-server:
	rsync -avr -e "ssh -l pi" --exclude 'venv/' ./* pi@${IP_ADDRESS}:/home/pi/camera/







