(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // ---------- site chrome ----------
  const burger = $('.burger'), menu = $('#menu');
  burger.addEventListener('click', () => { const o = menu.classList.toggle('open'); burger.setAttribute('aria-expanded', o); });
  $$('a[href="#top"]').forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    history.replaceState(null, '', '#top');
  }));
  menu.addEventListener('click', e => { if (e.target.closest('a')) { menu.classList.remove('open'); burger.setAttribute('aria-expanded', false); } });
  if ($('#yr')) $('#yr').textContent = new Date().getFullYear();


  // ---------- showcase ----------
  const tracksData = [
    ['Neon Nights', 'Synthwave · Night drive', 'Retro synthwave with pulsing bass and shimmering synths for a midnight drive', '#7c3aed,#ec4899', { genres: ['synthwave'], instruments: ['synthesizer', 'bass'], moods: ['dreamy'], bpm: 100 }],
    ['Summer Dreams', 'Pop · Upbeat', 'Sunny road-trip tune with bright guitars and handclaps', '#f59e0b,#ef4444', { genres: ['pop'], instruments: ['acoustic guitar', 'percussion'], moods: ['happy'], bpm: 112 }],
    ['Rainy Day Reverie', 'Ambient · Calm', 'Soft ambient piano with a slow, reflective feel', '#0ea5e9,#6366f1', { genres: ['ambient'], instruments: ['piano'], moods: ['calm'], bpm: 70 }],
    ['Dance All Night', 'House · Party', 'High-energy club track with a driving four-on-the-floor beat', '#ec4899,#8b5cf6', { genres: ['house'], instruments: ['synthesizer', 'drums'], moods: ['energetic'], bpm: 124 }],
    ['Echoes of Yesterday', 'Classical · Nostalgic', 'Emotional piano and strings, warm and wistful', '#14b8a6,#3b82f6', { genres: ['classical'], instruments: ['piano', 'strings'], moods: ['sad'], bpm: 72 }],
    ['Limitless Horizons', 'Cinematic · Epic', 'Rising orchestral anthem with big drums', '#22d3ee,#8b5cf6', { genres: ['cinematic'], instruments: ['strings', 'brass', 'drums'], moods: ['epic', 'uplifting'], bpm: 96 }]
  ];
  const wrap = $('#tracks');
  if (wrap) tracksData.forEach(([title, tags, text, g, preset], n) => {
    const [a, b] = g.split(',');
    const bars = Array.from({ length: 32 }, (_, i) => `<i style="--h:${Math.min(22 + Math.abs(Math.sin(i * 0.7 + n * 1.3) * 60) + (i * 7 + n * 13) % 18, 100)}%;--d:${(i % 8) * 0.07}s"></i>`).join('');
    const el = document.createElement('button');
    el.className = 'track rv'; el.type = 'button'; el.setAttribute('aria-label', `Use the prompt for ${title}`);
    el.innerHTML = `<div class="art" style="background:linear-gradient(135deg,${a},${b})" aria-hidden="true">${bars}</div>
      <div class="tb"><h3>${esc(title)}</h3><div class="tags">${tags.split(' · ').map(t => `<span>${t}</span>`).join('')}</div><p>${esc(text)}</p></div>`;
    el.addEventListener('click', () => {
      try { sessionStorage.setItem('vf-preset', JSON.stringify({ text, ...preset })); } catch {}
      location.href = 'create.html';
    });
    wrap.appendChild(el);
  });

  $$('.steps li, .card, .uc article, .plan, details:not(.lib)').forEach(e => e.classList.add('rv'));
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } }), { threshold: .12 });
    $$('.rv').forEach(e => io.observe(e));
  } else $$('.rv').forEach(e => e.classList.add('in'));
})();
