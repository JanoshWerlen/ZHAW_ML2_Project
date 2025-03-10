var istyping_icons = [
    "bi bi-chat aaa",
    "bi bi-chat-fill aaa",
    "bi bi-chat bbb",
    "bi bi-chat-fill bbb",
    "bi bi-chat ccc",
    "bi bi-chat-fill ccc",
    "bi bi-chat-dots aaa",
    "bi bi-chat-dots-fill aaa",
    "bi bi-chat-dots bbb",
    "bi bi-chat-dots-fill bbb",
    "bi bi-chat-dots ccc",
    "bi bi-chat-dots-fill ccc",
    "bi bi-chat-text aaa",
    "bi bi-chat-text-fill aaa",
    "bi bi-chat-text bbb",
    "bi bi-chat-text-fill bbb",
    "bi bi-chat-text ccc",
    "bi bi-chat-text-fill ccc",
    "bi bi-chat-quote aaa",
    "bi bi-chat-quote-fill aaa",
    "bi bi-chat-quote bbb",
    "bi bi-chat-quote-fill bbb",
    "bi bi-chat-quote ccc",
    "bi bi-chat-quote-fill ccc",
    "bi bi-chat-heart aaa",
    "bi bi-chat-heart-fill aaa",
    "bi bi-chat-heart bbb",
    "bi bi-chat-heart-fill bbb",
    "bi bi-chat-heart ccc",
    "bi bi-chat-heart-fill ccc",
    "bi bi-stopwatch aaa",
    "bi bi-stopwatch-fill aaa",
    "bi bi-stopwatch bbb",
    "bi bi-stopwatch-fill bbb",
    "bi bi-stopwatch ccc",
    "bi bi-stopwatch-fill ccc",
    "bi bi-hourglass-top aaa",
    "bi bi-hourglass-split aaa",
    "bi bi-hourglass-bottom aaa",
    "bi bi-hourglass-top bbb",
    "bi bi-hourglass-split bbb",
    "bi bi-hourglass-bottom bbb",
    "bi bi-hourglass-top ccc",
    "bi bi-hourglass-split ccc",
    "bi bi-hourglass-bottom ccc",
    "bi bi-hourglass",
    "bi bi-alarm aaa",
    "bi bi-alarm-fill aaa",
    "bi bi-alarm bbb",
    "bi bi-bell aaa",
    "bi bi-bell-fill aaa",
    "bi bi-bell bbb",
    "bi bi-alarm ccc",
    "bi bi-bell ccc",
    "bi bi-bell-fill bbb",
    "bi bi-emoji-smile aaa",
    "bi bi-emoji-smile-fill aaa",
    "bi bi-emoji-smile bbb",
    "bi bi-emoji-smile-fill bbb",
    "bi bi-emoji-neutral",
    "bi bi-emoji-neutral-fill",
    "bi bi-emoji-expressionless",
    "bi bi-emoji-expressionless-fill",
    "bi bi-emoji-dizzy",
    "bi bi-emoji-dizzy-fill",
    "bi bi-emoji-grimace",
    "bi bi-emoji-grimace-fill",
    "bi bi-emoji-astonished",
    "bi bi-emoji-astonished-fill",
    "bi bi-emoji-angry",
    "bi bi-emoji-angry-fill",
    "bi bi-emoji-frown",
    "bi bi-emoji-frown-fill",
    "bi bi-emoji-tear",
    "bi bi-emoji-tear-fill",
    "bi bi-heartbreak aaa",
    "bi bi-heartbreak-fill aaa",
    "bi bi-heartbreak bbb",
    "bi bi-heartbreak-fill bbb",
    "bi bi-heartbreak ccc",
    "bi bi-heartbreak-fill ccc",
    "bi bi-heart-pulse aaa",
    "bi bi-heart-pulse-fill aaa",
    "bi bi-heart-pulse bbb",
    "bi bi-heart-pulse-fill bbb",
    "bi bi-heart-pulse ccc",
    "bi bi-heart-pulse-fill ccc",
    "bi bi-heart-fill aaa",
    "bi bi-heart-half aaa",
    "bi bi-heart aaa",
    "bi bi-heart-fill bbb",
    "bi bi-heart-half bbb",
    "bi bi-heart bbb",
    "bi bi-heart-fill ccc",
    "bi bi-heart-half ccc",
    "bi bi-heart ccc",
    "bi bi-balloon-heart-fill aaa",
    "bi bi-balloon-heart aaa",
    "bi bi-balloon-heart-fill bbb",
    "bi bi-balloon-heart bbb",
];

var animate_istyping_interval = null;

function start_assistant_istyping_temp() {
    $('#messages').append(get_assistant_istyping_message());
    scroll_down();
    animate_istyping_interval = setInterval(function () {
        animate_istyping()
    }, 700);
    $('html,body').animate({
        scrollTop: 9999
    }, 'slow');
}


function scroll_down() {
    // $('html,body').animate({scrollTop: document.body.scrollHeight / 2}, 'fast');
    // $("section")[0].scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" });
    $('html,body').animate({
        scrollTop: document.body.scrollHeight
    }, 'fast');
}

function stop_assistant_istyping_temp() {
    if (animate_istyping_interval) {
        clearInterval(animate_istyping_interval);
        animate_istyping_interval = null;
    }
    $(".temporary").remove();
}

function animate_istyping() {
    let current_icon_index = istyping_icons.indexOf($("#istyping_icon").attr("class"));
    let new_icon_index = (current_icon_index + 1) % istyping_icons.length;

    $("#istyping_icon").removeClass(istyping_icons[current_icon_index]);
    $("#istyping_icon").addClass(istyping_icons[new_icon_index]);
}

function get_assistant_message(content) {
    return $("<div>").addClass("d-flex flex-row justify-content-start mb-4").append($("<i>").addClass("bi bi-emoji-sunglasses").attr("style", "font-size: 2rem;"), $("<div>").addClass("p-3 ms-3 border border-secondary").attr("style", "border-radius: 15px;").append($("<p>").addClass("small mb-0").html(content)));
}

function get_assistant_istyping_message() {
    return $("<div>").addClass("temporary d-flex flex-row justify-content-start mb-4").append($("<i>").addClass("bi bi-emoji-sunglasses").attr("style", "font-size: 2rem;"), $("<div>").addClass("p-3 ms-3 border border-secondary").attr("style", "border-radius: 15px;").append($("<p>").addClass("small mb-0").append($("<i>").addClass("bi bi-chat aaa").attr("id", "istyping_icon"))));
}

function get_user_message(content) {
    return $("<div>").addClass("d-flex flex-row justify-content-end mb-4").append($("<div>").addClass("p-3 me-3 border border-secondary").attr("style", "border-radius: 15px;").append($("<p>").addClass("small mb-0").text(content)), $("<i>").addClass("bi bi-person-bounding-box").attr("style", "font-size: 2rem;"));
}

function user_says() {
    const userInput = document.getElementById('user_says_input').value;
    const userFilter = document.querySelector('input[name="filter"]:checked').value;
    if (userInput.trim() === '') return;

    const messagesContainer = document.querySelector('.messages-container');
    const userMessageDiv = document.createElement('div');
    userMessageDiv.textContent = 'You: ' + userInput;
    userMessageDiv.classList.add('message', 'user-message');
    messagesContainer.appendChild(userMessageDiv);

    start_assistant_istyping_temp();

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput, filter_values: userFilter })
    })
        .then(response => response.json())
        .then(data => {
            stop_assistant_istyping_temp();
    
    
            const responseText = data.response;
    
            // Create a div for the bot's response
            const botMessageDiv = document.createElement('div');
            botMessageDiv.classList.add('message', 'bot-message');
    
            // Create a paragraph for the bot's response text
            const responseTextElement = document.createElement('p');
            responseTextElement.innerHTML = 'Bot: ' + responseText.replace(/\n/g, '<br>');
            botMessageDiv.appendChild(responseTextElement);
    
            // Add the bot's response to the messages container
            messagesContainer.appendChild(botMessageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

        });
    

    document.getElementById('user_says_input').value = '';

}

function validateForm() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    if (username.length > 8 || password.length > 8) {
        alert('Username and password must be no more than 8 characters long.');
        return false;
    }
    return true;
}

function logout(){
    fetch('logout')
}


function reset(event) {
    event.preventDefault();
    const sure = confirm("Willst Du den bisherigen Chatverlauf löschen?");
    if (sure) {
        const messagesContainer = document.querySelector('.messages-container');
        messagesContainer.innerHTML = '';
    }
}


function info(event) {
    event.preventDefault();
    alert("Role\n" + session.role + "\nContext\n" + session.context);
}