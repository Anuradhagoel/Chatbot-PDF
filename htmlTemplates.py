css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.chat-message {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 5px;
    display: flex;
    align-items: center; /* Vertically align items */
}

.chat-message.user {
    background-color: #2b313e;
    color: white;
    justify-content: flex-end; /* Push user message to the right */
    text-align: right;
}

.chat-message.bot {
    background-color: #475063;
    color: white;
    justify-content: flex-start; /* Push bot message to the left */
    text-align: left;
}

.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 10px; /* Space between avatar and message */
    margin-left: 10px; /* Space between avatar and message */
}

.chat-message.user .avatar {
    order: 2; /* Avatar after text for user */
    margin-left: 0; /* Remove extra left margin */
    margin-right: 10px; /* Keep right margin for separation */
}

.chat-message.bot .avatar {
    order: 1; /* Avatar before text for bot */
    margin-right: 0; /* Remove extra right margin */
    margin-left: 10px; /* Keep left margin for separation */
}

.chat-message .message-content {
    flex-grow: 1;
    word-wrap: break-word; /* Ensure long words wrap */
    overflow-wrap: break-word; /* Ensure long words wrap */
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/android-chrome-512x512.png" alt="Bot Avatar">
    </div>
    <div class="message-content">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="message-content">{{MSG}}</div>
    <div class="avatar">
        <img src="https://i.ibb.co/5Pj4c4r/user.png" alt="User Avatar">
    </div>
</div>
"""