document.querySelector('#upload-form').addEventListener('submit', sendRequest);


async function sendRequest(e) {
	e.preventDefault(); // Prevents the form from submitting the default way

	const fileInput = document.querySelector('#file');
	const file = fileInput.files[0];
	
	if (file) {
		const reader = new FileReader();
		reader.onload = function (e) {
			const image = document.createElement('img');
			image.src = e.target.result;
			image.width = 200;  // You can adjust the size as needed
			const displayDiv = document.getElementById('image-display');
			// Clear out any old images
			displayDiv.innerHTML = '';
			// Append the new image
			displayDiv.appendChild(image);
		};
		reader.readAsDataURL(file);
	}

	// Create a FormData object and append the file
	const formData = new FormData();
	formData.append('file', file);
	// Show the modal
	document.getElementById('loadingModal').style.display = 'block';
  
	try {
    // Make the request to the FastAPI server
    const response = await fetch('http://localhost:8501/cat-vs-dog/', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    // Hide the modal
    document.getElementById('loadingModal').style.display = 'none';

    // Display the classification
    const classification = data['classification'];
    const msg = document.getElementById('classification-msg');
    msg.innerText = classification;
    msg.style.display = 'block';
  } catch (error) {
    console.error('Error:', error);
    document.getElementById('errorModal').style.display = 'block';
  }
}