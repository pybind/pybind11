const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');

// Exercise the actual inline github-script without checking out or running PR code.
const workflow = fs.readFileSync(path.join(__dirname, '../workflows/intentional-ci.yml'), 'utf8');
const block = workflow.match(/          script: \|\n((?:            .*\n|\n)+)/)[1];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const dispatch = new AsyncFunction('github', 'context', 'core',
  block.split('\n').map(line => line.slice(12)).join('\n'));

async function run({command = '/ci full', pr = {}, commenter = 'rwgk', status} = {}) {
  const requests = [];
  const failures = [];
  const summaries = [];
  const summary = {
    addHeading(value) { summaries.push(value); return this; },
    addRaw() { return this; },
    addCodeBlock() { return this; },
    async write() {},
  };
  const github = {rest: {
    pulls: {async get() { return {data: {
      number: 123, state: 'open', user: {login: 'rwgk'},
      labels: [{name: 'ci:intentional'}],
      head: {sha: '1'.repeat(40)}, base: {sha: '2'.repeat(40)}, ...pr,
    }}; }},
    actions: {async createWorkflowDispatch(request) {
      if (status) throw Object.assign(new Error('API failure'), {status});
      requests.push(request);
    }},
  }};
  const context = {
    repo: {owner: 'pybind', repo: 'pybind11'}, issue: {number: 123},
    payload: {comment: {body: command, id: 456, user: {login: commenter}}},
  };
  await dispatch(github, context, {summary, setFailed(message) { failures.push(message); }});
  return {requests, failures, summaries};
}

for (const profile of ['full', 'quick']) {
  test(`dispatch ${profile} with frozen revisions to the fixed worker`, async () => {
    const {requests} = await run({command: `/ci ${profile}`});
    assert.equal(requests.length, 1);
    const {inputs, ...destination} = requests[0];
    assert.deepEqual(destination, {
      owner: 'pybind', repo: 'pybind11', workflow_id: 'intentional-ci.yml',
      ref: 'intentional-ci-experiment',
    });
    assert.deepEqual(JSON.parse(inputs.request), {
      pr: 123, profile, head_sha: '1'.repeat(40), base_sha: '2'.repeat(40), comment_id: 456,
    });
  });
}

for (const command of ['/ci', '/ci all', '/ci full extra', '> /ci full', '/ci full\nplease', '/ci full; echo unsafe']) {
  test(`ignore unrecognized command ${JSON.stringify(command)}`, async () => {
    assert.deepEqual((await run({command})).requests, []);
  });
}

for (const pr of [{state: 'closed'}, {user: {login: 'someone-else'}}, {labels: []}, {labels: [{name: 'ci:manual'}]}]) {
  test(`ignore PR outside the pilot: ${JSON.stringify(pr)}`, async () => {
    assert.deepEqual((await run({pr})).requests, []);
  });
}

test('other PR authors can request CI on their own opted-in PRs', async () => {
  const result = await run({commenter: 'contributor', pr: {user: {login: 'contributor'}}});
  assert.equal(result.requests.length, 1);
  assert.deepEqual(result.failures, []);
});

test('reject requests from someone other than the PR author', async () => {
  assert.deepEqual((await run({commenter: 'someone-else'})).requests, []);
});

test('help explains the pilot without dispatching', async () => {
  const result = await run({command: '/ci help'});
  assert.deepEqual(result.requests, []);
  assert.deepEqual(result.summaries, ['Intentional CI experiment']);
});

for (const status of [404, 422]) {
  test(`missing or incompatible worker fails clearly (${status})`, async () => {
    const result = await run({status});
    assert.deepEqual(result.requests, []);
    assert.match(result.failures[0], /intentional-ci-experiment/);
    assert.deepEqual(result.summaries, []);
  });
}

test('unexpected API errors propagate', async () => {
  await assert.rejects(run({status: 503}), /API failure/);
});
