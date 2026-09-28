const menu = document.querySelector('.menu-button');
menu?.addEventListener('click', () => {const open=document.querySelector('.sidebar').classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
document.querySelectorAll('.password-toggle').forEach(button=>button.addEventListener('click',()=>{
  const input=button.parentElement.querySelector('input');
  const showing=input.type==='password';
  input.type=showing?'text':'password';
  button.textContent=showing?'Hide':'Show';
  button.setAttribute('aria-pressed',String(showing));
}));
document.addEventListener('keydown', e => {if(e.key==='Escape'){document.querySelector('.sidebar')?.classList.remove('open');menu?.setAttribute('aria-expanded','false');}});
const upload=document.querySelector('#upload-form');
if(upload){const input=upload.querySelector('[type=file]'),summary=document.querySelector('#file-summary'),zone=document.querySelector('.dropzone');
input.addEventListener('change',()=>{summary.textContent=input.files.length ? Array.from(input.files).map(f=>f.name).join(', ') : 'No files selected';});
zone.addEventListener('dragover',()=>zone.classList.add('dragging'));for(const event of ['dragleave','drop'])zone.addEventListener(event,()=>zone.classList.remove('dragging'));
upload.addEventListener('submit',e=>{if(input.files.length>10){e.preventDefault();summary.textContent='Please select no more than 10 recordings.';return;}if(Array.from(input.files).some(f=>f.size>25*1024*1024)){e.preventDefault();summary.textContent='Each recording must be 25 MB or smaller.';return;}upload.querySelector('button').disabled=true;upload.querySelector('button').textContent='Analyzing audio…';document.querySelector('#processing').hidden=false;});}
window.addEventListener('pageshow',()=>{if(upload){upload.querySelector('button').disabled=false;upload.querySelector('button').textContent='Analyze recordings';document.querySelector('#processing').hidden=true;}});

document.addEventListener("click",e=>{if(!e.target.closest(".sidebar")&&!e.target.closest(".menu-button")){document.querySelector(".sidebar")?.classList.remove("open");menu?.setAttribute("aria-expanded","false");}});
