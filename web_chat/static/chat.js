const agentSelect = document.getElementById("agent-select");
const conversationEl = document.getElementById("conversation");
const form = document.getElementById("message-form");
const input = document.getElementById("message-input");
const newChatBtn = document.getElementById("new-chat-btn");

const state = {
  conversationId: null,
  agentId: null,
  isSending: false,
  agentReady: false,
  attachmentUrls: new Set(),
  conversationGeneration: 0,
};

function messageAttachments(message) {
  const attachments = message?.metadata?.attachments;
  if (Array.isArray(attachments)) return attachments;
  if (attachments != null) return [null];
  const content = message?.content;
  const parts = Array.isArray(content) ? content : content?.parts;
  if (!Array.isArray(parts)) return [];
  return parts.filter((part) =>
    part && ["file", "image", "audio", "video", "attachment"].includes(part.type)
  );
}

function messageText(message) {
  const content = message?.content;
  const parts = Array.isArray(content) ? content : content?.parts;
  const plain = Array.isArray(parts)
    ? parts.filter((part) => part?.type === "text" && part.format === "plain")
    : [];
  if (plain.length) return plain.map((part) => part.text || "").join("\n");
  return message?.raw_text || "";
}

function appendAttachment(list, attachment, index) {
  const item = document.createElement("li");
  const filename = attachment?.filename || attachment?.name;
  const label = typeof filename === "string" && filename ? filename : `File ${index + 1}`;
  try {
    if (!attachment || typeof attachment !== "object" || Array.isArray(attachment)) {
      throw new Error("invalid attachment metadata");
    }
    let href;
    let suppliedUrl = false;
    if (attachment.data != null) {
      if (typeof attachment.data !== "string") throw new Error("invalid file data");
      const decoded = atob(attachment.data);
      const bytes = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
      href = URL.createObjectURL(new Blob([bytes], {
        type: attachment.content_type || "application/octet-stream",
      }));
      state.attachmentUrls.add(href);
    } else if (typeof attachment.url === "string" && attachment.url) {
      const url = new URL(attachment.url, window.location.href);
      if (!["http:", "https:"].includes(url.protocol)) {
        throw new Error("unsupported download URL");
      }
      href = url.href;
      suppliedUrl = true;
    } else {
      throw new Error("file data or download URL was not supplied");
    }
    const link = document.createElement("a");
    link.href = href;
    if (suppliedUrl) link.target = "_blank";
    if (typeof filename === "string" && filename) link.download = filename;
    link.textContent = label;
    link.rel = "noopener noreferrer";
    item.appendChild(link);
  } catch (error) {
    item.className = "attachment-error";
    item.textContent = `${label} — Download unavailable: ${error.message || "invalid file data"}.`;
  }
  list.appendChild(item);
}

function appendMessage(role, text, attachments = []) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  div.appendChild(body);
  if (attachments.length) {
    const files = document.createElement("ul");
    files.className = "message-attachments";
    files.setAttribute("aria-label", "Attached files");
    attachments.forEach((attachment, index) => appendAttachment(files, attachment, index));
    div.appendChild(files);
  }
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
  state.conversationGeneration += 1;
  state.conversationId = null;
  state.agentReady = false;
  state.isSending = false;
  for (const url of state.attachmentUrls) URL.revokeObjectURL(url);
  state.attachmentUrls.clear();
  conversationEl.innerHTML = "";
  setStatus("Conversation cleared. Start chatting!");
});

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForConversationReady(conversationId, generation, maxAttempts) {
  let attempts = 0;
  while (generation === state.conversationGeneration &&
         (maxAttempts === undefined || attempts < maxAttempts)) {
    const res = await fetch(`/api/conversations/${conversationId}`);
    if (res.ok) {
      const data = await res.json();
      if (generation !== state.conversationGeneration) return false;
      if (data.status === "active") {
        state.agentReady = true;
        return true;
      }
    }
    attempts += 1;
    await delay(1000);
  }
  return false;
}

async function ensureConversation(generation) {
  if (generation !== state.conversationGeneration) return null;
  if (state.conversationId && state.agentReady) {
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
  if (generation !== state.conversationGeneration) return null;
  state.conversationId = data.id;
  state.agentReady = res.status === 201 && data.status === "active";
  if (!state.agentReady) {
    setStatus("Initializing agent, please wait…");
    await waitForConversationReady(state.conversationId, generation);
    if (generation !== state.conversationGeneration) return null;
    setStatus("");
  }
  return state.conversationId;
}

async function sendMessage(text, generation) {
  const conversationId = await ensureConversation(generation);
  if (conversationId === null) return null;
  while (generation === state.conversationGeneration) {
    const res = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (generation !== state.conversationGeneration) return null;
    if (res.status === 409) {
      setStatus("Agent is initializing, waiting for readiness…");
      await waitForConversationReady(conversationId, generation);
      if (generation !== state.conversationGeneration) return null;
      setStatus("Assistant is typing…");
      continue;
    }
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail?.detail || "Message failed");
    }
    return res.json();
  }
  return null;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isSending) return;
  const text = input.value.trim();
  if (!text) return;
  const generation = state.conversationGeneration;

  appendMessage("user", text);
  input.value = "";
  setStatus("Assistant is typing…");
  state.isSending = true;

  try {
    const response = await sendMessage(text, generation);
    if (generation !== state.conversationGeneration) return;
    const message = response?.agent_message;
    const assistantText = messageText(message);
    const attachments = messageAttachments(message);
    if (assistantText || attachments.length) {
      appendMessage("assistant", assistantText, attachments);
    }
  } catch (error) {
    if (generation === state.conversationGeneration) {
      console.error(error);
      setStatus(error.message || "Message failed");
    }
  } finally {
    if (generation === state.conversationGeneration) {
      state.isSending = false;
      if (state.agentReady) setStatus("");
      input.focus();
    }
  }
});

loadAgents().then(() => {
  setStatus("Select an agent and start chatting.");
});
