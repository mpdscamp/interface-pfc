import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Typography, TextField, Select, MenuItem, Button, FormControl,
  InputLabel, Paper, Alert, CircularProgress, Grid
} from '@mui/material';
import JobStatusTable from '../components/JobStatusTable';
import DatasetUploader from '../components/DatasetUploader';

interface Dataset { filename: string; size_mb: number }

export default function TrainingPage() {
  const [kind, setKind] = useState<'tabular' | 'llm'>('tabular');
  const [modelName, setModelName] = useState('distilbert-base-uncased');
  const [dataset, setDataset] = useState('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<{ ok:boolean; text:string } | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);

  const loadDatasets = useCallback(() =>
    fetch('/api/datasets').then(r => r.json()).then(setDatasets), []);

  useEffect(() => { loadDatasets(); }, [loadDatasets]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true); setMsg(null);
    const res = await fetch('/api/jobs/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind,
        ...(kind === 'llm' && { model_name: modelName }),
        dataset_filename: dataset
      })
    });
    const data = await res.json();
    setLoading(false);
    if (res.ok) {
      setMsg({ ok: true, text: 'Training job queued!' });
      setCurrentJobId(data.id);
      setDataset('');
    } else {
      setMsg({ ok: false, text: 'Failed to queue job' });
    }
  };

  const handleSave = async () => {
    if (!currentJobId) return;
    await fetch(`/api/jobs/${currentJobId}/checkpoint/save`, { method: 'POST' });
  };

  const handleStop = async () => {
    if (!currentJobId) return;
    await fetch(`/api/jobs/${currentJobId}/checkpoint/stop`, { method: 'POST' });
  };


  return (
    <Box>
      <Typography variant="h4" gutterBottom>Model Training</Typography>

      {/* uploader + form side-by-side */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DatasetUploader onUploaded={loadDatasets}/>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p:3 }}>
            <form onSubmit={submit}>
              <Box sx={{ display:'flex', flexDirection:'column', gap:2 }}>
                <FormControl fullWidth>
                  <InputLabel>Model Family</InputLabel>
                  <Select value={kind} label="Model Family"
                          onChange={e => setKind(e.target.value as any)}>
                    <MenuItem value="tabular">Tabular (Random Forest)</MenuItem>
                    <MenuItem value="llm">LLM (DistilBERT)</MenuItem>
                  </Select>
                </FormControl>

                {kind === 'llm' && (
                  <TextField
                    label="Model Name"
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    required
                    fullWidth
                  />
                )}

                <FormControl fullWidth required>
                  <InputLabel>Dataset</InputLabel>
                  <Select value={dataset} label="Dataset"
                          onChange={e => setDataset(e.target.value)}>
                    {datasets.map(d => (
                      <MenuItem key={d.filename} value={d.filename}>{d.filename}</MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {msg && <Alert severity={msg.ok?'success':'error'}>{msg.text}</Alert>}

                <Button type="submit" variant="contained"
                  disabled={loading || !dataset}
                  startIcon={loading ? <CircularProgress size={20}/> : null}>
                  Start Training
                </Button>
              </Box>
            </form>
          </Paper>
        </Grid>
      </Grid>

      <Typography variant="h5" sx={{ mt:3 }}>Job Status</Typography>
      {kind === 'llm' && currentJobId && (
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button variant="outlined" onClick={handleSave}>
            Save Checkpoint
          </Button>
          <Button variant="contained" color="error" onClick={handleStop}>
            Stop Training
          </Button>
        </Box>
      )}
      <JobStatusTable />
    </Box>
  );
}
