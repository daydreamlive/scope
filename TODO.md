# Daydream Scope — AI Persona TODO

## In Progress

- [~] Define product direction and update CLAUDE.md with persona vision
- [~] Research character consistency techniques (persistent latent conditioning, reference image anchoring, IP-Adapter)

## Phase 1: Foundation — Action Schema & Interpreter

- [ ] Design structured action/expression schema (Pydantic models for `{ action, expression, dialogue, intensity }`)
- [ ] Define action vocabulary — enumerate supported actions (wave, sit, stand, nod, turn, walk, etc.) and expressions (smile, frown, laugh, confused, neutral, etc.)
- [ ] Build action interpreter module (`scope.core.persona.action_interpreter`) — LLM call that maps free-text user input to structured action directives
- [ ] Write tests for action interpreter (edge cases: ambiguous commands, multiple actions, unknown actions)
- [ ] Decide on LLM provider/model for action interpretation (local vs API, latency budget)

## Phase 2: Conversation Layer

- [ ] Build conversation manager (`scope.core.persona.conversation`) — maintains chat history, persona system prompt, personality definition
- [ ] Design persona personality config schema (name, backstory, voice style, behavioral constraints)
- [ ] Create chat API endpoint (`server/chat.py`) — WebSocket or SSE for real-time back-and-forth
- [ ] Wire conversation manager → action interpreter → pipeline manager flow
- [ ] Add character state tracker (`scope.core.persona.state`) — tracks current pose, expression, activity for coherent transitions

## Phase 3: Persona Video Pipeline

- [ ] Scaffold persona pipeline directory (`core/pipelines/persona/`)
- [ ] Implement persona pipeline `__call__()` accepting structured action directives (not raw text)
- [ ] Integrate character consistency module — evaluate approaches: IP-Adapter, reference image conditioning, LoRA identity, latent anchoring
- [ ] Implement smooth action transitions (don't jump-cut between poses; interpolate or blend)
- [ ] Benchmark frame latency — must stay real-time (<100ms per frame target)
- [ ] Test character identity drift over long sessions (>5 min continuous generation)

## Phase 4: Frontend Chat UI

- [ ] Design chat interface component (message history + text input)
- [ ] Integrate chat UI alongside or replacing the timeline/prompt editor
- [ ] Show character status indicator (current action/expression)
- [ ] Display real-time video stream next to chat panel
- [ ] Add action shortcut buttons (quick-fire common actions like wave, smile, nod)

## Phase 5: Polish & Future

- [ ] Audio/TTS integration — persona speaks responses, lip-sync with generated video
- [ ] Emotion inference from conversation context (auto-set expression without explicit user command)
- [ ] Multi-character support (stretch goal)
- [ ] Persona gallery — prebuilt characters users can select
- [ ] Export/record conversation sessions as video files

## Completed

- [x] Initial project setup (LongLive pipeline working with WebRTC streaming)
- [x] Update CLAUDE.md with AI persona product direction and architectural plan
- [x] Create TODO.md for session continuity

## Decisions & Notes

- **Separation of concerns is critical**: Conversational AI layer (LLM, chat, action interpretation) must be fully decoupled from video generation layer (pipeline). They communicate via the structured action schema only.
- **Character consistency is the hardest unsolved problem** — research this early, prototype multiple approaches before committing.
- **Latency budget**: Action interpreter LLM call + video generation must fit within ~200ms total for real-time feel. Consider pre-generating idle animations to fill gaps.
- **Existing LongLive pipeline stays intact** — persona pipeline is a new pipeline registered alongside it, not a replacement.
