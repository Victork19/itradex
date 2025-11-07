//assets/settings.js
const PREMIUM_KEY = 'dropimus_premium';
const STORAGE_KEY = 'dropimus_vault_v1';
const WALLET_KEY = 'dropimus_wallet';
const SIGNATURE_KEY = 'dropimus_signature';
const TWITTER_KEY = 'dropimus_twitter';
const MESSAGE = 'Dropimus wallet connection signature\n' + window.location.origin + '\n' + Date.now();

function isPremium() {
  return localStorage.getItem(PREMIUM_KEY) === 'true';
}

function getConnectedWallet() {
  return localStorage.getItem(WALLET_KEY) || null;
}

function getSignature() {
  return localStorage.getItem(SIGNATURE_KEY) || null;
}

function getConnectedTwitter() {
  return localStorage.getItem(TWITTER_KEY) || null;
}

function updateWalletDisplay() {
  const wallet = getConnectedWallet();
  const display = document.getElementById('walletDisplay');
  const subDisplay = document.getElementById('subWalletDisplay');
  const btn = document.getElementById('connectWalletBtn');
  if (wallet) {
    const shortened = wallet.slice(0, 6) + '…' + wallet.slice(-4);
    display.innerHTML = `<span class="wallet-address">${shortened}</span>`;
    subDisplay.textContent = shortened;
    btn.textContent = 'Connected';
    btn.classList.add('btn-connected');
  } else {
    display.innerHTML = '<span class="wallet-not-connected">Not connected</span>';
    subDisplay.textContent = 'Not connected';
    btn.textContent = 'Connect';
    btn.classList.remove('btn-connected');
  }
}

function updateTwitterDisplay() {
  const twitter = getConnectedTwitter();
  const display = document.getElementById('twitterDisplay');
  const btn = document.getElementById('connectTwitterBtn');
  if (twitter) {
    display.innerHTML = `<span class="twitter-handle">@${twitter}</span>`;
    btn.textContent = 'Connected';
    btn.classList.add('btn-connected');
  } else {
    display.innerHTML = '<span class="twitter-not-connected">Not connected</span>';
    btn.textContent = 'Connect';
    btn.classList.remove('btn-connected');
  }
}

async function connectAndSign() {
  if (typeof window.ethereum !== 'undefined') {
    try {
      // Request account access
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
      const address = accounts[0];

      // Check if already signed
      if (getConnectedWallet() === address && getSignature()) {
        return address;
      }

      // Sign message
      const signature = await window.ethereum.request({
        method: 'personal_sign',
        params: [MESSAGE, address],
      });

      // TODO: Send to FastAPI backend for verification and storage
      // await fetch('/api/wallet/connect', { method: 'POST', body: JSON.stringify({ address, signature }) });

      // For demo, store locally
      localStorage.setItem(WALLET_KEY, address);
      localStorage.setItem(SIGNATURE_KEY, signature);

      return address;
    } catch (error) {
      console.error('Wallet connection failed:', error);
      if (error.code === 4001) {
        throw new Error('User rejected the connection request');
      }
      throw error;
    }
  } else {
    throw new Error('MetaMask or compatible wallet not installed');
  }
}

function promptUpgrade(context = '') {
  if (confirm(`Pro feature unlocked with SuperGrok subscription. ${context ? context + '. ' : ''}Redirect to subscribe?`)) {
    window.open('https://dropimus.com', '_blank');
  }
}

function showModal(modalId){ 
  const sidebar = document.querySelector('.sidebar');
  sidebar?.classList.remove('open');
  document.getElementById('overlay').classList.remove('hidden'); 
  document.getElementById(modalId).classList.remove('hidden'); 
}

function hideAllModals(){ 
  const sidebar = document.querySelector('.sidebar');
  sidebar?.classList.remove('open');
  document.getElementById('overlay').classList.add('hidden'); 
  document.querySelectorAll('[id$="Modal"]').forEach(m => m.classList.add('hidden')); 
}

// Mobile menu
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.querySelector('.sidebar');
const overlay = document.getElementById('modalRoot');
mobileMenuBtn?.addEventListener('click', () => {
  sidebar?.classList.add('open');
  overlay.innerHTML = '<div style="position:absolute;inset:0;background:rgba(0,0,0,0.55)"></div>';
});
overlay.addEventListener('click', (e) => {
  if (e.target === overlay.firstChild) {
    sidebar?.classList.remove('open');
    overlay.innerHTML = '';
  }
});

// Wallet connection
function openWalletModal() {
  showModal('walletModal');
  document.getElementById('confirmWallet').style.display = 'none';
}

document.getElementById('connectWalletBtn').addEventListener('click', openWalletModal);
document.getElementById('changeWalletBtn').addEventListener('click', openWalletModal);
document.getElementById('closeWalletX').addEventListener('click', hideAllModals);
document.getElementById('cancelWallet').addEventListener('click', hideAllModals);

document.getElementById('connectMetamask').addEventListener('click', async () => {
  const btn = document.getElementById('connectMetamask');
  btn.classList.add('loading');
  try {
    const address = await connectAndSign();
    updateWalletDisplay();
    hideAllModals();
    alert(`Wallet connected: ${address.slice(0, 6)}…${address.slice(-4)}`);
  } catch (error) {
    console.error('Connection error:', error);
    if (error.message.includes('User rejected')) {
      alert('Connection denied by user.');
    } else if (error.message.includes('not installed')) {
      alert('Please install MetaMask or a compatible wallet to continue.');
    } else {
      alert('Connection failed. Please try again.');
    }
  } finally {
    btn.classList.remove('loading');
  }
});

document.getElementById('connectWalletConnect').addEventListener('click', () => {
  alert('WalletConnect integration coming soon.');
});

// Twitter connection
document.getElementById('connectTwitterBtn').addEventListener('click', () => showModal('twitterModal'));
document.getElementById('closeTwitterX').addEventListener('click', hideAllModals);
document.getElementById('cancelTwitter').addEventListener('click', hideAllModals);

document.getElementById('connectTwitter').addEventListener('click', async () => {
  const btn = document.getElementById('connectTwitter');
  btn.classList.add('loading');
  try {
    // TODO: Redirect to FastAPI OAuth endpoint for Twitter
    // window.location.href = '/api/twitter/oauth';

    // For demo, mock
    if (confirm('Redirect to X for authentication? (Mock)')) {
      const mockHandle = 'hunter_eth'; // Mock or fetch from OAuth callback
      // TODO: Save to backend via API
      // await fetch('/api/twitter/connect', { method: 'POST', body: JSON.stringify({ handle: mockHandle }) });

      localStorage.setItem(TWITTER_KEY, mockHandle);
      updateTwitterDisplay();
      hideAllModals();
      alert(`Twitter connected: @${mockHandle}`);
    }
  } catch (error) {
    console.error('Twitter connection failed:', error);
    alert('Connection failed. Please try again.');
  } finally {
    btn.classList.remove('loading');
  }
});

// Password edit
document.getElementById('editPasswordBtn').addEventListener('click', () => showModal('passwordModal'));
document.getElementById('closePasswordX').addEventListener('click', hideAllModals);
document.getElementById('cancelPassword').addEventListener('click', hideAllModals);

document.getElementById('savePassword').addEventListener('click', () => {
  const current = document.getElementById('currentPasswordInput').value;
  const newPass = document.getElementById('newPasswordInput').value;
  const confirm = document.getElementById('confirmPasswordInput').value;

  if (!current || !newPass || !confirm) {
    alert('All fields required.');
    return;
  }
  if (newPass !== confirm) {
    alert('Passwords do not match.');
    return;
  }
  if (newPass.length < 8) {
    alert('Password must be at least 8 characters.');
    return;
  }
  // TODO: Send to FastAPI backend for update
  // await fetch('/api/user/password', { method: 'PUT', body: JSON.stringify({ current, new: newPass }) });

  // Mock save password
  localStorage.setItem('dropimus_password_hash', btoa(newPass)); // Simple mock
  hideAllModals();
  alert('Password updated successfully.');
  // Clear inputs
  document.getElementById('currentPasswordInput').value = '';
  document.getElementById('newPasswordInput').value = '';
  document.getElementById('confirmPasswordInput').value = '';
});

// Subscription management
document.getElementById('manageSubscriptionBtn').addEventListener('click', () => showModal('subscriptionModal'));
document.getElementById('closeSubscriptionX').addEventListener('click', hideAllModals);

// Plan upgrade buttons
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('plan-upgrade-btn')) {
    const plan = e.target.dataset.plan;
    openUpgradeModal(plan);
  }
});

// Token selection in upgrade modal
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('token-option')) {
    e.target.parentNode.querySelectorAll('.token-option').forEach(opt => opt.classList.remove('selected'));
    e.target.classList.add('selected');
    const newToken = e.target.dataset.token;
    const usdPrice = planPrices[currentUpgradePlan];
    if (newToken === 'ETH') {
      const ethAmount = (usdPrice / window.ethPrice || 4000).toFixed(4);
      document.querySelector('.upgrade-cost').textContent = `Cost: ${ethAmount} ETH (≈${usdPrice} USD)`;
    } else {
      document.querySelector('.upgrade-cost').textContent = `${usdPrice} USDC`;
    }
  }
});

// Upgrade Modal Functions
let currentUpgradePlan = '';
const planPrices = { 'Starter': 5, 'Core': 9, 'Pro': 17 }; // in USD equiv
const planIds = { 'Starter': 1, 'Core': 2, 'Pro': 3 };

async function getEthPrice() {
  try {
    const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd');
    if (!response.ok) throw new Error('Failed to fetch ETH price');
    const data = await response.json();
    if (!data.ethereum || !data.ethereum.usd) throw new Error('Invalid price data');
    return data.ethereum.usd;
  } catch (error) {
    console.warn('Failed to fetch ETH price, using fallback:', error);
    return 4000; // Approximate fallback for Oct 2025
  }
}

async function openUpgradeModal(plan) {
  currentUpgradePlan = plan;
  const usdPrice = planPrices[plan];
  if (!usdPrice) {
    alert('Invalid plan selected');
    return;
  }
  const networkName = window.NETWORK_MODE === 'sepolia' ? 'Sepolia' : 'Base';
  document.querySelector('.upgrade-title').textContent = `Upgrade to ${plan}`;
  document.querySelector('.upgrade-cost').textContent = `${usdPrice} USDC`;
  document.querySelector('#upgradeConfirmModal .mb-4 p').textContent = `Choose token (on ${networkName}):`;
  document.querySelector('.gas-estimate').textContent = `Gas estimate: ~0.0001 ETH (${networkName})`;
  showModal('upgradeConfirmModal');

  // Fetch ETH price for display
  try {
    const ethPrice = await getEthPrice();
    window.ethPrice = ethPrice;
  } catch (error) {
    console.error('ETH price fetch error:', error);
    window.ethPrice = 4000;
  }
}
document.getElementById('closeUpgradeX').addEventListener('click', hideAllModals);
document.getElementById('cancelUpgrade').addEventListener('click', hideAllModals);

// Confirm payment
document.querySelector('.confirm-btn').addEventListener('click', async () => {
  const btn = document.querySelector('.confirm-btn');
  btn.classList.add('loading');
  try {
    if (!getConnectedWallet()) {
      alert('Please connect your wallet first.');
      return;
    }
    const selectedTokenOption = document.querySelector('.token-option.selected');
    const token = selectedTokenOption ? selectedTokenOption.dataset.token : 'USDC';
    const userId = localStorage.getItem('user_id') || 'demo_user';
    const planId = planIds[currentUpgradePlan];
    if (!planId) {
      throw new Error('Invalid plan ID');
    }
    const txHash = await window.pay(userId, planId, token); // Use window.pay
    localStorage.setItem(PREMIUM_KEY, 'true');
    localStorage.setItem('current_plan', currentUpgradePlan);
    updateProBadges();
    document.getElementById('currentPlan').textContent = `You’re on: ${currentUpgradePlan}`;
    hideAllModals();
    alert(`Upgraded to ${currentUpgradePlan}! Tx: ${txHash.slice(0, 10)}...`);
  } catch (error) {
    console.error('Payment error:', error);
    alert('Payment failed: ' + error.message);
  } finally {
    btn.classList.remove('loading');
  }
});

function confirmTransaction() {
  // Deprecated, using event listener above
}

// Cancel Modal
function openCancelModal() {
  showModal('cancelConfirmModal');
}
document.getElementById('closeCancelX').addEventListener('click', hideAllModals);
document.getElementById('cancelCancel').addEventListener('click', hideAllModals);
document.getElementById('confirmCancel').addEventListener('click', () => {
  if (confirm('Are you sure? This action cannot be undone.')) {
    // TODO: Call FastAPI to cancel
    // await fetch('/api/subscription/cancel', { method: 'POST' });
    localStorage.setItem(PREMIUM_KEY, 'false');
    localStorage.removeItem('current_plan');
    updateProBadges();
    document.getElementById('currentPlan').textContent = `You’re on: Free`;
    alert('Subscription canceled.');
    hideAllModals();
  }
});

// Backup functionality
document.getElementById('createBackupBtn').addEventListener('click', () => showModal('backupModal'));
document.getElementById('createBackup').addEventListener('click', () => {
  const password = document.getElementById('backupPassword').value;
  if (!password) { 
    alert('Password required'); 
    return; 
  }
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const encrypted = CryptoJS.AES.encrypt(JSON.stringify(data), password).toString();
    const blob = new Blob([encrypted], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; 
    a.download = `dropimus_backup_${new Date().toISOString().slice(0,10)}.enc`; 
    document.body.appendChild(a); 
    a.click(); 
    a.remove(); 
    URL.revokeObjectURL(url);
    hideAllModals();
    document.getElementById('backupPassword').value = '';
  } catch (error) {
    console.error('Backup creation failed:', error);
    alert('Failed to create backup.');
  }
});
document.getElementById('closeBackupX').addEventListener('click', hideAllModals);
document.getElementById('cancelBackup').addEventListener('click', hideAllModals);

// Restore functionality
document.getElementById('restoreFile').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const password = prompt('Enter password to decrypt');
  if (!password) {
    e.target.value = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const decrypted = CryptoJS.AES.decrypt(ev.target.result, password).toString(CryptoJS.enc.Utf8);
      if (!decrypted) throw new Error('Decryption failed');
      const data = JSON.parse(decrypted);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      alert('Restored successfully!');
      e.target.value = '';
    } catch (err) {
      console.error('Restore failed:', err);
      alert('Failed to decrypt or invalid file.');
      e.target.value = '';
    }
  };
  reader.onerror = () => {
    alert('File read error.');
    e.target.value = '';
  };
  reader.readAsText(file);
});

// Account edit functionality
document.getElementById('editNameBtn').addEventListener('click', () => {
  const currentName = localStorage.getItem('dropimus_display_name') || 'hunter.eth';
  document.getElementById('newNameInput').value = currentName;
  showModal('accountEditModal');
});
document.getElementById('closeAccountX').addEventListener('click', hideAllModals);
document.getElementById('cancelName').addEventListener('click', hideAllModals);
document.getElementById('saveName').addEventListener('click', () => {
  const newName = document.getElementById('newNameInput').value.trim();
  if (!newName) { 
    alert('Name required'); 
    return; 
  }
  try {
    document.getElementById('currentName').childNodes[0].textContent = newName;
    // TODO: Save to FastAPI backend
    // await fetch('/api/user/name', { method: 'PUT', body: JSON.stringify({ name: newName }) });

    // Mock save to localStorage or API
    localStorage.setItem('dropimus_display_name', newName);
    hideAllModals();
    // Optional: Update sidebar account display
    const sidebarName = document.getElementById('sidebarAccountName');
    if (sidebarName) sidebarName.textContent = newName;
  } catch (error) {
    console.error('Name update failed:', error);
    alert('Failed to update name.');
  }
});

// Preferences toggle
document.getElementById('autoRefreshToggle').addEventListener('change', (e) => {
  const isEnabled = e.target.checked;
  // TODO: Save to FastAPI backend
  // await fetch('/api/user/notifications', { method: 'PUT', body: JSON.stringify({ enabled: isEnabled }) });

  localStorage.setItem('dropimus_auto_refresh', isEnabled);
  console.log(`Auto-refresh ${isEnabled ? 'enabled' : 'disabled'}`);
});

// Logout
document.querySelector('.logout-btn').addEventListener('click', () => {
  if (confirm('Are you sure you want to log out?')) {
    localStorage.clear();
    alert('Logged out successfully.');
    // window.location.href = 'login.html'; // Uncomment for real app
  }
});

// Pro Badge Visibility
function updateProBadges() {
  const isPro = isPremium();
  const badges = ['proBadge', 'sidebarProBadge'];
  badges.forEach(badgeId => {
    const badge = document.getElementById(badgeId);
    if (badge) {
      badge.style.display = isPro ? 'inline-block' : 'none';
    }
  });
}

// Highlight active nav
function highlightActiveNav() {
  const path = window.location.pathname.split('/').pop();
  const dashboardBtn = document.getElementById('dashboardBtn');
  const trackerBtn = document.getElementById('trackerBtn');
  const settingsBtn = document.getElementById('settingsBtn');
  // Reset
  if (dashboardBtn) dashboardBtn.style.background = '';
  if (trackerBtn) trackerBtn.style.background = '';
  if (settingsBtn) settingsBtn.style.background = 'rgba(255,255,255,0.03)';
}

// Initial load
const currentPlan = localStorage.getItem('current_plan') || 'Free';
document.getElementById('currentPlan').textContent = `You’re on: ${currentPlan}`;
if (currentPlan === 'Free') {
  document.getElementById('planExpires').style.display = 'none';
}
highlightActiveNav();
updateProBadges();
updateWalletDisplay();
updateTwitterDisplay();
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    hideAllModals();
    sidebar?.classList.remove('open');
  }
});

// Expose functions to window for HTML onclick
window.openCancelModal = openCancelModal;

// Magnet effect script
(function(){
  const buttons = Array.from(document.querySelectorAll('.nl-magnet'));
  if(!buttons.length) return;

  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const touchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0 || window.innerWidth < 680;

  buttons.forEach(btn => {
    const bg = btn.querySelector('.nlbg');
    const DAMP = 0.18;            // smoothing factor (higher = snappier, less laggy)
    const RETURN_DAMP = 0.25;
    const DEFAULT_MAX = 12;       // max px movement (desktop)
    const DEFAULT_THRESHOLD = 110; // px (how far pointer can be to start effect)
    let max = DEFAULT_MAX;
    let threshold = DEFAULT_THRESHOLD;

    if(window.innerWidth < 900) { max = 9; threshold = 90; }
    if(window.innerWidth < 640) { max = 6; threshold = 64; }

    // on pure touch / small screen, don't run follow behavior — just highlight on touch/focus
    const enableFollow = !touchDevice && !prefersReduced;

    let target = {x:0,y:0}, current = {x:0,y:0}, rafId = null;
    let isHovering = false;

    function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
    function lerp(a,b,t){ return a + (b-a) * t; }

    function computeTargetFromPointer(clientX, clientY){
      const r = btn.getBoundingClientRect();
      const cx = r.left + r.width/2;
      const cy = r.top + r.height/2;
      const dx = clientX - cx;
      const dy = clientY - cy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if(dist > threshold){
        target.x = 0; target.y = 0;
        btn.classList.remove('is-near');
        return;
      }
      const t = 1 - (dist / threshold); // 0..1 (closer = larger)
      // invert direction slightly so button moves toward pointer but not too aggressively
      target.x = clamp((dx / (dist || 1)) * max * t * 0.9, -max, max);
      target.y = clamp((dy / (dist || 1)) * max * t * 0.75, -max, max);
      if (isHovering) {
        target.y -= 2;
      }
      btn.classList.add('is-near');
    }

    function onPointerMove(e){
      const cx = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] && e.touches[0].clientX) || (window.innerWidth/2);
      const cy = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0] && e.touches[0].clientY) || (window.innerHeight/2);
      computeTargetFromPointer(cx, cy);
      startLoop();
    }

    function onPointerLeave(){
      target.x = 0; target.y = 0;
      btn.classList.remove('is-near');
      startLoop();
    }

    function tick(){
      // lerp toward target (smoother)
      const damp = isHovering ? DAMP : RETURN_DAMP;
      current.x = lerp(current.x, target.x, damp);
      current.y = lerp(current.y, target.y, damp);

      // Apply transform
      btn.style.transform = `translate3d(${current.x}px, ${current.y}px, 0)`;

      // subtle background parallax
      if(bg){
        const bgX = current.x * 0.35;
        const bgY = current.y * 0.22;
        const scale = 1 + Math.min(0.06, Math.abs(current.x+current.y)/300);
        bg.style.transform = `translate(calc(-50% + ${bgX}px), calc(-50% + ${bgY}px)) scale(${scale})`;
        bg.style.opacity = String(0.38 + Math.min(0.45, Math.abs(current.x+current.y)/60));
      }

      // stop loop if both current and target are near zero
      if(Math.abs(current.x - target.x) < 0.05 && Math.abs(current.y - target.y) < 0.05 && Math.abs(target.x) < 0.01 && Math.abs(target.y + (isHovering ? 2 : 0)) < 0.01){
        cancelAnimationFrame(rafId); rafId = null;
        // keep transform cleared for crisp rendering
        btn.style.transform = '';
        if(bg) bg.style.transform = '';
        return;
      }
      rafId = requestAnimationFrame(tick);
    }

    function startLoop(){
      if(rafId) return;
      rafId = requestAnimationFrame(tick);
    }

    // Event binding
    if(enableFollow){
      const onMouseEnter = (e) => {
        isHovering = true;
        onPointerMove(e);
      };
      btn.addEventListener('mousemove', onPointerMove, {passive:true});
      btn.addEventListener('mouseenter', onMouseEnter, {passive:true});
      btn.addEventListener('mouseleave', () => {
        isHovering = false;
        onPointerLeave();
      });
      // touch: simulate with touchmove while touching button (some browsers)
      btn.addEventListener('touchstart', (ev)=>{ if(ev.touches && ev.touches[0]) onPointerMove(ev.touches[0]); }, {passive:true});
      btn.addEventListener('touchmove', (ev)=>{ if(ev.touches && ev.touches[0]) onPointerMove(ev.touches[0]); }, {passive:true});
      btn.addEventListener('touchend', onPointerLeave, {passive:true});
    } else {
      // fallback: highlight on focus/press only — avoids heavy pointer processing on mobile
      btn.addEventListener('touchstart', ()=>{ btn.classList.add('is-near'); }, {passive:true});
      btn.addEventListener('touchend', ()=>{ btn.classList.remove('is-near'); }, {passive:true});
    }

    // Keyboard accessibility: show focused state but don't move
    btn.addEventListener('focus', ()=>{ btn.classList.add('is-focused'); btn.classList.add('is-near'); }, true);
    btn.addEventListener('blur', ()=>{ btn.classList.remove('is-focused'); btn.classList.remove('is-near'); }, true);

    // clean up on page hide
    document.addEventListener('visibilitychange', ()=>{ if(document.hidden && rafId){ cancelAnimationFrame(rafId); rafId = null; }});
  });
})();