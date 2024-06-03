// Track selected option
let selectedOption = 'option1';  // default selection
console.log("ahmad")
// Function to update selection
function updateSelection(selected) {
    const options = document.querySelectorAll('li[data-value]');
    options.forEach(option => {
        if (option.getAttribute('data-value') === selected) {
            option.classList.replace('bg-gray-300', 'bg-blue-500');
            option.classList.replace('text-black', 'text-white');
        } else {
            option.classList.replace('bg-blue-500', 'bg-gray-300');
            option.classList.replace('text-white', 'text-black');
        }
    });
    selectedOption = selected;
}

// Event listeners for each option
document.getElementById('option1').addEventListener('click', () => updateSelection('option1'));
document.getElementById('option2').addEventListener('click', () => updateSelection('option2'));

function sendSearch() {
    const query = document.getElementById('searchQuery').value;
    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, option: selectedOption })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('results').innerHTML = data.map(item =>
           item
        ).join('');
    })
    .catch(error => console.error('Error:', error));
}