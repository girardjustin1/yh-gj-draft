/* ==================================================================================
   OFFLINE ENGINE — a faithful port of the Python decision engine.

   GitHub Pages runs no Python, and this is where the tool is actually used, so the
   parts that decide a pick are ported rather than omitted: opponent roster needs,
   roster-aware survival, the Monte Carlo simulation, and two-pick expected value.

   Porting means two implementations, which can drift. That risk is handled rather than
   accepted: tests/test_js_parity.py runs this file under node against the Python
   originals on fixed inputs and fails if they disagree. The deterministic parts must
   match to 1e-9; the Monte Carlo, which cannot share numpy's RNG, must agree within
   sampling error.

   Everything expensive and slow-moving — projections, VBD, replacement, tiers,
   floor/ceiling — is still computed in Python at build time and baked into board.json.
   What runs here is only what depends on live draft state.
   ================================================================================== */
const Engine = (() => {
  const OFFENSE = ["QB","RB","WR","TE"];
  const FLEX_ELIGIBILITY = {
    FLEX:["RB","WR","TE"], WRRB_FLEX:["RB","WR"], REC_FLEX:["WR","TE"],
    SUPERFLEX:["QB","RB","WR","TE"], OP:["QB","RB","WR","TE"],
  };
  const DECAY_SCALE = 1.25;        // availability.DECAY_SCALE
  const DEFAULT_ADP_SD = 12.0;     // availability.DEFAULT_ADP_SD
  const CANDIDATE_WINDOW = 60;     // availability.CANDIDATE_WINDOW
  const SIM_POOL_SIZE = 220;       // simulator.SIM_POOL_SIZE
  const NEED_STRENGTH = 1.6;       // opponent_needs.NEED_STRENGTH
  const MIN_POSITION_PROB = 0.02;  // opponent_needs.MIN_POSITION_PROBABILITY

  /* ---- snake maths: draft/snake.py ---- */
  function forward(rd, L){
    if(L.draft_type === "linear") return true;
    const std = rd % 2 === 1;
    return (L.third_round_reversal && rd >= 3) ? !std : std;
  }
  const roundFor = (overall, L) => Math.floor((overall - 1) / L.teams) + 1;
  function slotFor(overall, L){
    const within = ((overall - 1) % L.teams) + 1;
    return forward(roundFor(overall, L), L) ? within : L.teams - within + 1;
  }
  function pickNumber(rd, slot, L){
    return (rd - 1) * L.teams + (forward(rd, L) ? slot : L.teams - slot + 1);
  }
  const label = (overall, L) =>
    `${roundFor(overall,L)}.${String(((overall-1)%L.teams)+1).padStart(2,"0")}`;
  const picksForSlot = (slot, L) =>
    Array.from({length:L.rounds}, (_,i) => pickNumber(i+1, slot, L));
  function slotsBetween(startOverall, endOverall, L){
    const out=[], lo=Math.max(1,startOverall+1), hi=Math.min(L.teams*L.rounds,endOverall-1);
    for(let p=lo;p<=hi;p++) out.push([p, slotFor(p,L)]);
    return out;
  }

  /* ---- normal CDF, for the ADP-only baseline ---- */
  function erf(x){                       // Abramowitz & Stegun 7.1.26
    const sgn = x<0?-1:1; x=Math.abs(x);
    const t = 1/(1+0.3275911*x);
    return sgn*(1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t
      +0.254829592)*t*Math.exp(-x*x));
  }
  const normCdf = (x) => 0.5*(1+erf(x/Math.SQRT2));
  function adpSurvival(adp, sd, nextPick){          // availability.adp_survival
    if(adp==null) return 0.5;
    const spread = Math.max(sd ?? DEFAULT_ADP_SD, 1);
    return 1 - normCdf((nextPick - adp)/spread);
  }

  /* ---- unfilled_starters: opponent_needs.unfilled_starters ---- */
  function unfilledStarters(L, positionCounts){
    const counts = positionCounts || {};
    const remaining = {}, need = {};
    for(const p of OFFENSE) remaining[p] = counts[p] || 0;
    for(const p of OFFENSE){
      const required = L.dedicated[p] || 0;
      const used = Math.min(remaining[p], required);
      remaining[p] -= used;
      need[p] = Math.max(0, required - used);
    }
    for(const [slotName, count] of Object.entries(L.flex_counts||{})){
      const eligible = (FLEX_ELIGIBILITY[slotName]||[]).filter(p=>OFFENSE.includes(p));
      if(!eligible.length) continue;
      const spare = eligible.reduce((s,p)=>s+remaining[p],0);
      const unfilled = Math.max(0, count - spare);
      const consumed = Math.min(spare, count);
      for(const p of eligible){
        if(spare>0) remaining[p] -= consumed*(remaining[p]/spare);
        need[p] += unfilled/eligible.length;
      }
    }
    return need;
  }

  /* ---- market_prior: read the positional mix off the board itself ---- */
  function marketPrior(board, lo, hi){
    const span = Math.max(hi-lo, 1);
    for(const widen of [0, span, span*3]){
      const win = board.filter(p => p.adp!=null && p.adp>=lo-widen && p.adp<=hi+widen
        && OFFENSE.includes(p.position));
      if(win.length>=6){
        const mix={}; for(const p of OFFENSE) mix[p]=0;
        for(const p of win) mix[p.position] += 1/win.length;
        return mix;
      }
    }
    const flat={}; for(const p of OFFENSE) flat[p]=1/OFFENSE.length;
    return flat;
  }

  /* ---- position_probabilities ---- */
  function positionProbabilities(L, positionCounts, prior, roundNumber){
    const need = unfilledStarters(L, positionCounts);
    const depth = Math.min(1, roundNumber / Math.max(L.rounds*0.6, 1));
    const strength = 1 + NEED_STRENGTH*depth;
    const weights={}; let total=0;
    for(const p of OFFENSE){
      const base = Math.max(prior[p]||0, MIN_POSITION_PROB);
      weights[p] = base * (1 + strength*Math.min(need[p]||0, 2));
      total += weights[p];
    }
    const out={};
    if(total<=0){ for(const p of OFFENSE) out[p]=1/OFFENSE.length; return out; }
    for(const p of OFFENSE) out[p]=weights[p]/total;
    return out;
  }

  /* ---- opponent_needs: per-pick intent for everyone before our next turn ---- */
  function opponentNeeds(L, order, myCurrent, myNext, board){
    const upcoming = slotsBetween(myCurrent, myNext, L);
    if(!upcoming.length) return [];
    // Opponent rosters, reconstructed from pick order exactly as DraftState.rosters().
    const rosters = {};
    order.forEach((key, i) => {
      const s = slotFor(i+1, L);
      (rosters[s] = rosters[s] || []).push(key);
    });
    const lo = Math.min(...upcoming.map(([o])=>o));
    const hi = Math.max(...upcoming.map(([o])=>o));
    const prior = marketPrior(board, lo, hi);
    const posOf = {}; for(const p of board) posOf[p.player_key]=p.position;

    const projected = {}, needs = [];
    for(const [overall, slot] of upcoming){
      const counts = {};
      for(const k of (rosters[slot]||[])){ const pos=posOf[k]; if(pos) counts[pos]=(counts[pos]||0)+1; }
      for(const pos of (projected[slot]||[])) counts[pos]=(counts[pos]||0)+1;
      const probs = positionProbabilities(L, counts, prior, roundFor(overall, L));
      const top = OFFENSE.reduce((a,b)=>probs[b]>probs[a]?b:a, OFFENSE[0]);
      (projected[slot]=projected[slot]||[]).push(top);
      needs.push({overall, slot, probabilities:probs});
    }
    return needs;
  }

  /* ---- survival_probabilities: the roster-aware analytic model ---- */
  function survivalProbabilities(available, needs, limit=CANDIDATE_WINDOW){
    if(!needs.length || !available.length) return {estimates:{}, losses:{}};
    const pool = available.slice(0, Math.max(limit,120)).map(p=>({
      key:p.player_key, position:p.position,
      adp: p.adp!=null?p.adp:250,
      spread: Math.max(p.adp_sd ?? DEFAULT_ADP_SD, 2),
    }));
    const alive={}; for(const e of pool) alive[e.key]=1;
    const losses={};
    for(const need of needs){
      let reference = Infinity;
      for(const e of pool) if(alive[e.key]>0.05 && e.adp<reference) reference=e.adp;
      if(!isFinite(reference)) reference = need.overall;
      const weights={}; let total=0;
      for(const e of pool){
        const so_far = alive[e.key];
        if(so_far<=1e-6) continue;
        const decay = DECAY_SCALE*e.spread;
        const desirability = Math.exp(-Math.max(0, e.adp-reference)/decay);
        const w = so_far * desirability * (need.probabilities[e.position]||0);
        weights[e.key]=w; total+=w;
      }
      if(total<=0) continue;
      for(const e of pool){
        const w = weights[e.key]; if(w===undefined) continue;
        const taken = w/total;
        losses[e.position]=(losses[e.position]||0)+taken;
        alive[e.key] *= 1-taken;
      }
    }
    const estimates={};
    for(const e of pool.slice(0,limit)) estimates[e.key]=Math.min(1,Math.max(0,alive[e.key]));
    return {estimates, losses};
  }

  /* ---- Monte Carlo: simulator.simulate_to_next_pick ----
     numpy's PCG64 cannot be reproduced here, so this uses a seeded mulberry32 and the
     parity test compares distributions within sampling error rather than exactly. */
  function mulberry32(seed){
    let a = seed >>> 0;
    return function(){
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function simulate(available, needs, opts={}){
    const iterations = opts.iterations ?? 1500;
    const seed = opts.seed ?? 20260831;
    const bestAvailableRate = opts.bestAvailableRate ?? 0.35;
    const candidates = opts.candidates ?? [];
    if(!needs.length || !available.length)
      return {iterations:0, picks:0, survival:{}, losses:{}, twoPick:{}};

    const pool = available.slice(0, SIM_POOL_SIZE);
    const n = pool.length;
    const adp = pool.map(p=>p.adp!=null?p.adp:250);
    const decay = pool.map(p=>DECAY_SCALE*Math.max(p.adp_sd ?? DEFAULT_ADP_SD, 2));
    const value = pool.map(p=>p.player_score ?? 0);
    const posIndex = pool.map(p=>OFFENSE.indexOf(p.position));
    const rng = mulberry32(seed);

    const alive = new Uint8Array(iterations*n).fill(1);
    const losses={}; for(const p of OFFENSE) losses[p]=0;

    for(const need of needs){
      const needVec = OFFENSE.map(p=>Math.max(need.probabilities[p]||0,1e-4));
      const flat = OFFENSE.map(()=>1/OFFENSE.length);
      for(let s=0;s<iterations;s++){
        const base = s*n;
        const useFlat = rng() < bestAvailableRate;
        const weightsByPos = useFlat ? flat : needVec;
        let reference = Infinity;
        for(let i=0;i<n;i++) if(alive[base+i] && adp[i]<reference) reference=adp[i];
        if(!isFinite(reference)) reference = need.overall;
        let total=0; const w=new Float64Array(n);
        for(let i=0;i<n;i++){
          if(!alive[base+i]) continue;
          const d = Math.exp(-Math.max(0, adp[i]-reference)/decay[i]);
          const pos = posIndex[i];
          w[i] = d * (pos>=0 ? weightsByPos[pos] : 1e-4);
          total += w[i];
        }
        if(total<=0) continue;
        let draw = rng()*total, chosen=-1;
        for(let i=0;i<n;i++){ draw-=w[i]; if(draw<=0 && w[i]>0){ chosen=i; break; } }
        if(chosen<0) for(let i=n-1;i>=0;i--) if(w[i]>0){ chosen=i; break; }
        if(chosen<0) continue;
        alive[base+chosen]=0;
        const pos = OFFENSE[posIndex[chosen]];
        if(pos) losses[pos] += 1/iterations;
      }
    }

    const survival={};
    for(let i=0;i<n;i++){
      let c=0; for(let s=0;s<iterations;s++) c+=alive[s*n+i];
      survival[pool[i].player_key]=c/iterations;
    }

    /* two-pick expected value: simulator._two_pick_values */
    const order = Array.from({length:n},(_,i)=>i).sort((a,b)=>value[b]-value[a]);
    const twoPick={};
    for(const key of candidates.slice(0, opts.twoPickCandidates ?? 8)){
      const idx = pool.findIndex(p=>p.player_key===key);
      if(idx<0) continue;
      const rank = order.indexOf(idx);
      let sum=0; const samples=[];
      for(let s=0;s<iterations;s++){
        const base=s*n;
        let first=-1, second=-1;
        for(const oi of order){ if(alive[base+oi]){ if(first<0) first=oi; else {second=oi; break;} } }
        const pick = (first===idx) ? second : first;
        const v = pick>=0 ? value[pick] : 0;
        sum+=v; samples.push(v);
      }
      samples.sort((a,b)=>a-b);
      const q=(p)=>samples[Math.min(samples.length-1,Math.floor(p*samples.length))];
      const mean=sum/iterations;
      twoPick[key]={player_key:key, value_now:value[idx], expected_next_value:mean,
        combined:value[idx]+mean, low:q(0.10), high:q(0.90)};
      void rank;
    }
    return {iterations, picks:needs.length, survival, losses, twoPick};
  }

  /* ---- marginal lineup value: analytics/lineup_value, in VBD ---- */
  function bestLineup(L, players){
    const slots=L.lineup_slots;
    const ded=slots.filter(s=>!FLEX_ELIGIBILITY[s]);
    const flx=slots.filter(s=>FLEX_ELIGIBILITY[s]);
    const pool=[...players].sort((a,b)=>b[1]-a[1]);
    const used=new Set(); let total=0;
    for(const s of [...ded,...flx]){
      const elig=FLEX_ELIGIBILITY[s]||[s];
      for(let i=0;i<pool.length;i++){
        if(used.has(i)||!elig.includes(pool[i][0])) continue;
        used.add(i); total+=pool[i][1]; break;
      }
    }
    return total;
  }

  /* ---- team strength: service.PickAnalysis.team_strength ----
     Coverage of the lineup you have to fill, not a comparison against other teams —
     that version reported "1 of 1" for everything as soon as you recorded only your own
     picks, which is the normal way to use this. */
  function requiredSlots(L, position){
    let n = (L.dedicated||{})[position] || 0;
    for(const [name,count] of Object.entries(L.flex_counts||{})){
      const eligible = FLEX_ELIGIBILITY[name] || [];
      if(eligible.includes(position)) n += count / eligible.length;
    }
    return Math.max(1, Math.round(n));
  }

  function teamStrength(L, picks, mySlot, board, losses, _unused){
    const vbd={}, pos={};
    for(const p of board){ vbd[p.player_key]=p.vbd??0; pos[p.player_key]=p.position; }
    const myKeys = picks.filter(p=>p.mine).map(p=>p.k);

    // Best available at each position, to price what could still fill an empty slot.
    const byPos={};
    for(const p of board){
      if(p.vbd==null) continue;
      (byPos[p.position]=byPos[p.position]||[]).push(p.vbd);
    }
    for(const v of Object.values(byPos)) v.sort((a,b)=>b-a);

    // Opponent rosters, for the optional league line only.
    const opponents={};
    picks.forEach((pick,i)=>{
      if(pick.mine) return;
      const owner=`s${slotFor(i+1,L)}`;
      const position=pos[pick.k]; if(!position) return;
      (opponents[owner]=opponents[owner]||{});
      opponents[owner][position]=(opponents[owner][position]||0)+(vbd[pick.k]||0);
    });

    const rows=[];
    for(const position of OFFENSE){
      const required = requiredSlots(L, position);
      const mine = myKeys.filter(k=>pos[k]===position)
        .map(k=>vbd[k]||0).sort((a,b)=>b-a);
      const starters = mine.slice(0, required);
      const have = starters.reduce((s,v)=>s+Math.max(v,0),0);
      const gap = Math.max(0, required-starters.length);
      const fillers = (byPos[position]||[]).slice(0, gap);
      const target = have + fillers.reduce((s,v)=>s+Math.max(v,0),0);
      const coverage = target>0 ? have/target*100 : (gap===0?100:0);
      const drain = Math.min(1,(losses[position]||0)/4);
      const priority = 0.70*(100-coverage) + 0.30*drain*100;

      const others = Object.values(opponents).map(t=>t[position]||0).sort((a,b)=>a-b);
      let league=null;
      if(others.length){
        const below = others.filter(v=>v<have).length;
        league = {rank: others.length+1-below, teams: others.length+1,
          percentile: Math.round(below/others.length*100),
          median: others[Math.floor(others.length/2)]};
      }
      rows.push({position, required, filled:starters.length,
        have_value:+have.toFixed(1), target_value:+target.toFixed(1),
        coverage:Math.round(coverage), priority:Math.round(Math.min(100,Math.max(0,priority))),
        expected_gone_before_next_pick:+(losses[position]||0).toFixed(1), league});
    }
    rows.sort((a,b)=>b.priority-a.priority);
    return {positions:rows, top_priority:rows.length?rows[0].position:null,
      has_league_comparison:Object.keys(opponents).length>0,
      opponent_teams:Object.keys(opponents).length};
  }

  return {OFFENSE, FLEX_ELIGIBILITY, teamStrength, requiredSlots, forward, roundFor, slotFor, pickNumber, label,
    picksForSlot, slotsBetween, adpSurvival, normCdf, unfilledStarters, marketPrior,
    positionProbabilities, opponentNeeds, survivalProbabilities, simulate, bestLineup};
})();
if (typeof module !== "undefined") module.exports = Engine;
