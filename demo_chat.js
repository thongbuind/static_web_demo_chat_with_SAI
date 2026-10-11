const FALLBACK_HISTORY_FILE = "chat_history/00_tong_quan.json"
const chatEl = document.getElementById("chat")
const demoNotice = document.getElementById("demo-notice")

let sidebarOpen = false
let currentSessionId = null
let savedSessions = []

function closeDemoNotice() {
  if (!demoNotice || demoNotice.classList.contains("is-hidden")) return
  demoNotice.classList.add("is-hidden")
  demoNotice.setAttribute("aria-hidden", "true")
  document.getElementById("sidebar-open-btn")?.focus()
}

demoNotice?.querySelectorAll("[data-modal-close]").forEach(element => {
  element.addEventListener("click", closeDemoNotice)
})

document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeDemoNotice()
})

requestAnimationFrame(() => {
  document.getElementById("demo-notice-confirm")?.focus()
})

function toggleTheme() {
  window.SAITheme?.toggle()
}

function toggleSidebar() {
  sidebarOpen = !sidebarOpen
  document.getElementById("sidebar").classList.toggle("closed", !sidebarOpen)
  document.getElementById("main").classList.toggle("sidebar-open", sidebarOpen)
  document.getElementById("sidebar-open-btn").classList.toggle("hidden", sidebarOpen)
}

function showHistoryStatus(message) {
  const list = document.getElementById("history-list")
  list.innerHTML = ""

  const status = document.createElement("div")
  status.className = "history-empty"
  status.textContent = message
  list.appendChild(status)

}

function isValidSession(value) {
  return value && typeof value === "object" && Array.isArray(value.messages)
}

function normalizeSession(value, fallbackId) {
  if (!isValidSession(value)) return null
  const firstUserMessage = value.messages.find(message => message.role === "user")?.text
  return {
    ...value,
    id: String(value.id || fallbackId),
    title: String(value.title || firstUserMessage || "Cuộc trò chuyện").slice(0, 80),
  }
}

async function readSession(url, fallbackId) {
  const response = await fetch(url, { cache: "no-store" })
  if (!response.ok) throw new Error(`Không thể đọc ${url}`)
  return normalizeSession(await response.json(), fallbackId)
}

async function discoverHistoryUrls() {
  const response = await fetch("chat_history/manifest.json", { cache: "no-store" })
  if (!response.ok) throw new Error("Không thể đọc manifest lịch sử")

  const manifest = await response.json()
  if (!manifest || !Array.isArray(manifest.files)) {
    throw new Error("Manifest lịch sử không hợp lệ")
  }

  return manifest.files
    .filter(file => typeof file === "string" && /^[a-zA-Z0-9_.-]+\.json$/i.test(file))
    .map(file => new URL(file, response.url).href)
}

function renderHistoryList() {
  const list = document.getElementById("history-list")
  list.innerHTML = ""

  if (!savedSessions.length) {
    showHistoryStatus("Không có lịch sử chat.")
    return
  }

  savedSessions.forEach(session => {
    const item = document.createElement("button")
    item.type = "button"
    item.className = `history-item${session.id === currentSessionId ? " active" : ""}`
    item.dataset.id = session.id

    const title = document.createElement("span")
    title.className = "history-title"
    title.textContent = session.title
    item.appendChild(title)
    item.onclick = () => loadSession(session.id)
    list.appendChild(item)
  })

}

function normalizeAIText(text) {
  const normalized = String(text ?? "").replace(/\r\n?/g, "\n").replace(/\\r\\n|\\n|\\r/g, "\n")
  let result = ""
  let capitalizeNext = true

  for (let index = 0; index < normalized.length; index++) {
    const character = normalized[index]
    if (/\p{L}/u.test(character)) {
      result += capitalizeNext ? character.toLocaleUpperCase("vi-VN") : character
      capitalizeNext = false
      continue
    }

    result += character
    if (/\p{N}/u.test(character)) capitalizeNext = false
    else if (character === "\n") capitalizeNext = true
    else if (character === ".") {
      const previous = normalized[index - 1] || ""
      const next = normalized[index + 1] || ""
      if (!(/\p{N}/u.test(previous) && /\p{N}/u.test(next))) capitalizeNext = true
    }
  }
  return result
}

function renderAIText(element, text) {
  element.replaceChildren()
  const boldPattern = /\*\*([\s\S]+?)\*\*/g
  let cursor = 0
  let match

  while ((match = boldPattern.exec(text)) !== null) {
    element.appendChild(document.createTextNode(text.slice(cursor, match.index)))
    const strong = document.createElement("strong")
    strong.textContent = match[1]
    element.appendChild(strong)
    cursor = match.index + match[0].length
  }
  element.appendChild(document.createTextNode(text.slice(cursor)))
}

function feedbackHTML() {
  return `<button class="feedback-btn" data-type="like" title="Hữu ích"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg></button><button class="feedback-btn" data-type="dislike" title="Không hữu ích"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg></button>`
}

function bindFeedbackButtons(container) {
  container.querySelectorAll(".feedback-btn").forEach(button => {
    button.onclick = () => {
      const wasActive = button.classList.contains("active")
      container.querySelectorAll(".feedback-btn").forEach(item => item.classList.remove("active", "liked", "disliked"))
      if (!wasActive) button.classList.add("active", button.dataset.type === "like" ? "liked" : "disliked")
    }
  })
}

function renderMessage(role, text, model) {
  if (role === "ai" && model) {
    const modelLabel = document.createElement("div")
    modelLabel.className = "model-label-row"
    modelLabel.textContent = String(model).replace(/\.pt$/i, "")
    chatEl.appendChild(modelLabel)
  }

  const row = document.createElement("div")
  row.className = `message-row ${role}`
  const bubble = document.createElement("div")
  bubble.className = "bubble glass-bubble glass-content"

  if (role === "ai") {
    const avatar = document.createElement("div")
    avatar.className = "ai-avatar"
    avatar.textContent = "SAI"
    row.appendChild(avatar)

    const body = document.createElement("div")
    body.className = "ai-msg-body"
    renderAIText(bubble, normalizeAIText(text))
    body.appendChild(bubble)

    const feedback = document.createElement("div")
    feedback.className = "feedback-row"
    feedback.innerHTML = feedbackHTML()
    bindFeedbackButtons(feedback)
    body.appendChild(feedback)
    row.appendChild(body)
  } else {
    bubble.textContent = String(text ?? "")
    row.appendChild(bubble)
  }

  chatEl.appendChild(row)
}

function loadSession(id) {
  const session = savedSessions.find(item => item.id === id)
  if (!session) return

  currentSessionId = session.id
  document.title = `${session.title} — SAI`
  chatEl.innerHTML = ""
  session.messages.forEach(message => renderMessage(message.role, message.text, message.model))
  chatEl.scrollTop = 0
  renderHistoryList()
}

async function loadInitialHistory() {
  showHistoryStatus("Đang tải lịch sử…")
  chatEl.innerHTML = '<div class="chat-loading">Đang tải cuộc trò chuyện…</div>'

  try {
    const urls = await discoverHistoryUrls()
    const results = await Promise.allSettled(urls.map((url, index) => readSession(url, `saved-${index}`)))
    savedSessions = results
      .filter(result => result.status === "fulfilled" && result.value)
      .map(result => result.value)
  } catch {
    try {
      const fallback = await readSession(FALLBACK_HISTORY_FILE, "first-chat")
      savedSessions = fallback ? [fallback] : []
    } catch {
      savedSessions = []
    }
  }

  if (savedSessions.length) {
    loadSession(savedSessions[0].id)
  } else {
    chatEl.innerHTML = '<div class="chat-loading">Không thể tải lịch sử chat.</div>'
    showHistoryStatus("Không thể tải lịch sử chat.")
  }
}

loadInitialHistory()
