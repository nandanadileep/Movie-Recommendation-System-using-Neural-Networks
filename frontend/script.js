document.getElementById("get-recommendations").addEventListener("click", async () => {
    const textarea = document.getElementById("favorite-movies");
    const movieInput = textarea.value;
    const movieList = movieInput.split(",").map(m => m.trim()).filter(m => m);

    if (movieList.length === 0) {
        alert("Please enter at least one movie.");
        return;
    }

    try {
        const response = await fetch("https://movies-recommendation-uzpq.onrender.com", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ favorite_movies: movieList })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        const recommendationsList = document.getElementById("recommendations-list");
        recommendationsList.innerHTML = "";

        data.recommendations.forEach(movie => {
            const li = document.createElement("li");
            li.textContent = `${movie.title} (${movie.genres}) — Score: ${movie.predicted_score.toFixed(2)}`;
            recommendationsList.appendChild(li);
        });

    } catch (err) {
        console.error(err);
        alert("Error connecting to the backend. Make sure it's deployed!");
    }
});
