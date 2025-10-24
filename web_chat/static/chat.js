const agentSelect = document.getElementById("agent-select");
const conversationEl = document.getElementById("conversation");
const form = document.getElementById("message-form");
const input = document.getElementById("message-input");
const newChatBtn = document.getElementById("new-chat-btn");

const state = {
  conversationId: null,
  agentId: null,
  isSending: false,
};

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  conversationEl.appendChild(div);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function setStatus(text) {
  let statusEl = conversationEl.querySelector(".status");
  if (!statusEl && text) {
    statusEl = document.createElement("div");
    statusEl.className = "status";
    conversationEl.appendChild(statusEl);
  }
  if (statusEl) {
    if (text) {
      statusEl.textContent = text;
    } else {
      statusEl.remove();
    }
  }
}

async function loadAgents() {
  try {
    const res = await fetch("/api/agents");
    if (!res.ok) {
      throw new Error("Failed to load agents");
    }
    const agents = await res.json();
    agentSelect.innerHTML = "";
    agents.forEach((agent) => {
      const option = document.createElement("option");
      option.value = agent.id;
      option.textContent = `${agent.name}`;
      agentSelect.appendChild(option);
    });
    state.agentId = agents[0]?.id || null;
  } catch (error) {
    console.error(error);
    agentSelect.innerHTML = "<option>Unavailable</option>";
  }
}

agentSelect.addEventListener("change", (event) => {
  state.agentId = event.target.value;
});

newChatBtn.addEventListener("click", () => {
  state.conversationId = null;
  conversationEl.innerHTML = "";
  setStatus("Conversation cleared. Start chatting!");
});

async function ensureConversation() {
  if (state.conversationId) {
    return state.conversationId;
  }
  if (!state.agentId) {
    throw new Error("Select an agent first");
  }
  const res = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent_id: state.agentId }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || "Failed to create conversation");
  }
  const data = await res.json();
  state.conversationId = data.id;
  return state.conversationId;
}

async function sendMessage(text) {
  const conversationId = await ensureConversation();
  const res = await fetch(`/api/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || "Message failed");
  }
  return res.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isSending) return;
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";
  setStatus("Assistant is typing…");
  state.isSending = true;

  try {
    const response = await sendMessage(text);
    const assistantText = response?.agent_message?.raw_text || "";
    if (assistantText) {
      appendMessage("assistant", assistantText);
    }
  } catch (error) {
    console.error(error);
    appendMessage("assistant", `⚠️ ${error.message}`);
  } finally {
    state.isSending = false;
    setStatus("");
    input.focus();
  }
});

loadAgents().then(() => {
  setStatus("Select an agent and start chatting.");
});
