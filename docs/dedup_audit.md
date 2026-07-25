# Dedup audit — dropped near-duplicate pairs (method=greedy, threshold=3)

- total dropped images: **3583**
- clusters formed: **2031**, largest: **38** images, mean dropped/cluster: **1.8**
- dropped from large groups (>10 imgs): **459** (12.8% of drops)
- drops involving military_vessel: **555** (15.5% of drops)
- distance-to-representative histogram: {0: 1114, 2: 2469}

Each dropped image A is within threshold of a KEPT representative B (greedy leader dedup — no transitive chaining, so every `dist` <= threshold). Open A vs B.

| pair | dist | military | A (dropped) | B (kept) | hashA | hashB |
|---|---:|:---:|---|---|---|---|
| pair00_a.jpg / pair00_b.jpg | 0 | yes | shiprsimagenet__1441__2027_0_bmp.rf.d59cf9aa49c4d7f14f08b6cedae8e261 | military_ships__1441__2027_0_bmp.rf.297a19ba8576251a9b58131ecd8c9133 | c3b8df1b1918d1c3 | c3b8df1b1918d1c3 |
| pair01_a.jpg / pair01_b.jpg | 0 | yes | shiprsimagenet__100001627_bmp.rf.633cb67adacb6f058f6af1b0a5be79c5 | military_ships__100001627_bmp.rf.e252e27e2bf637fd9e68b5fdd7a76ca9 | ffea60b01b193c46 | ffea60b01b193c46 |
| pair02_a.jpg / pair02_b.jpg | 0 | yes | shiprsimagenet__100001411_bmp.rf.df90b317eac70c47cabd2df832c68f45 | military_ships__100001411_bmp.rf.dd5be32b7abcda411884d4adc6608baf | cce7339c863389c6 | cce7339c863389c6 |
| pair03_a.jpg / pair03_b.jpg | 0 | no | shiprsimagenet__1158__0_1798_bmp.rf.ea5ed88a721f1ef5977ea1cdaab18895 | military_ships__1158__0_1798_bmp.rf.9c55622e920edeaae02fd528e3485d97 | 9a15a5e2ce5cc07b | 9a15a5e2ce5cc07b |
| pair04_a.jpg / pair04_b.jpg | 0 | no | seaships__000402_jpg.rf.659e8dfd104ef73189e6406696b6f0ad | seaships__000401_jpg.rf.87bbc076f41c82cbdf5cb0cb9251a23d | 81962f766d0a95ba | 81962f766d0a95ba |
| pair05_a.jpg / pair05_b.jpg | 0 | no | seaships__005765_jpg.rf.91d9a026fe37bc49c09cc18a51cd8b5d | seaships__005566_jpg.rf.151dff22c2d93ea6aec021ba73953258 | c0c0d0f3b73f2d2c | c0c0d0f3b73f2d2c |
| pair06_a.jpg / pair06_b.jpg | 0 | no | shiprsimagenet__1443__2027_1884_bmp.rf.c316596476babbe334de0fbb577e88c2 | military_ships__1443__2027_1884_bmp.rf.8b9f9ac43350a87190d2b4accfecccc7 | a24ee93fa35c1c2a | a24ee93fa35c1c2a |
| pair07_a.jpg / pair07_b.jpg | 2 | no | seaships__000665_jpg.rf.bfe30856b26336b2bf6e9d19e91bfbf4 | seaships__000664_jpg.rf.549c71c1b1c70f1cdd0aded624221301 | d590be817c09a4df | d590be017e09a4df |
| pair08_a.jpg / pair08_b.jpg | 2 | no | seaships__002616_jpg.rf.f786e62bce1a7c9e4a336af0501cb985 | seaships__002480_jpg.rf.017934355fa09eddd6465d6b5f854cb8 | 85763ec3d44b813e | 85763ec1d44bc13e |
| pair09_a.jpg / pair09_b.jpg | 2 | no | seaships__006661_jpg.rf.1a2d5ee34a127ee574b0dd4cc772529e | seaships__002854_jpg.rf.ea0baa3eb8f413a92212e1016ad97823 | 817e1ec3fc6b8056 | 81761ec3fc6b8156 |
| pair10_a.jpg / pair10_b.jpg | 2 | no | seaships__000727_jpg.rf.18d89a4cd5cb7d43686a22c0ef98b6b1 | seaships__000728_jpg.rf.694d00b11953ea0c5a2371cba7b5a1c9 | a4b1d8196437f6c6 | a4b1d819643fe6c6 |
| pair11_a.jpg / pair11_b.jpg | 2 | no | seaships__001283_jpg.rf.5e55fe78b9cde7d4c8c3b7694bf4a791 | seaships__001168_jpg.rf.9ab4a1f732b213e6c7c5b8a8d8440af8 | 9485b2f47f0f1d81 | 9485b2b47f4f1d81 |
| pair12_a.jpg / pair12_b.jpg | 2 | no | seaships__002038_jpg.rf.4290c5066686b76324c1db2fe2b9875c | seaships__002004_jpg.rf.aecd7aafdd952dfdea2b4b1d52b10497 | c1760fc1f66b813c | c1761fc1f46b813c |
| pair13_a.jpg / pair13_b.jpg | 2 | no | seaships__003673_jpg.rf.2c2ea5de4f74aff57c0df27ffc8c5764 | seaships__003500_jpg.rf.8ef6c4ce09deecafdc6f1e5f967acdb3 | c57a6841fc0bc1b7 | c5726843fc0bc1b7 |
| pair14_a.jpg / pair14_b.jpg | 2 | no | seaships__004339_jpg.rf.fbea97b5a91a9cc7ea1cdac2ff4f9220 | seaships__004230_jpg.rf.f64d8c398f253e329f9f79c0396062b6 | 84761ecbf66b811c | 84760ecbf66b813c |
| pair15_a.jpg / pair15_b.jpg | 2 | no | seaships__004937_jpg.rf.7a24d12089c8e00bedd8532e5ff9321f | seaships__002230_jpg.rf.e0eefed8f4a1400821631e73d7b0deb0 | 90766bc39c1bc176 | 807e6bc39c1bc176 |
| pair16_a.jpg / pair16_b.jpg | 2 | no | seaships__005377_jpg.rf.7f769b55afe96371cca0439b10b876ee | seaships__005373_jpg.rf.e427b1137df276e4bcae3daa2f52431e | dcdc75d59111998a | d8dc75d591119b8a |
| pair17_a.jpg / pair17_b.jpg | 2 | no | seaships__005805_jpg.rf.09383cc0f1f9d47723a0a7311708f309 | seaships__005804_jpg.rf.14cfd19cf546887d0bd577cfc9534893 | d0c0c2d7b73f2d0c | c0c0d2d7b73f2d0c |
| pair18_a.jpg / pair18_b.jpg | 2 | no | seaships__006145_jpg.rf.5c70c43b077018aced8bfccc6e397c7e | seaships__006143_jpg.rf.e3ab7f8f7128aa33ead38ae3aac6e630 | c0c0d28397af3dec | c0c0d68397ad3dec |
| pair19_a.jpg / pair19_b.jpg | 2 | no | seaships__006659_jpg.rf.098ec9c544a6bbb43131e78ffeaad237 | seaships__004302_jpg.rf.dd57e4c2fb8e6a7029fca363d2519887 | 84763e4bd649c13d | 80767e4bd649c13d |
| pair20_a.jpg / pair20_b.jpg | 2 | yes | shiprsimagenet__000867_bmp.rf.7b6e79d0b41d0b565259d615f60a8433 | military_ships__000867_bmp.rf.85336ee2ea2b904d8567bb4952ccaebc | a701b4e727b1585b | a701b4e727b2585b |
| pair21_a.jpg / pair21_b.jpg | 2 | yes | shiprsimagenet__004647_bmp.rf.64e0629e8ebbde18cf2c3eab3e24ea47 | military_ships__004647_bmp.rf.41276ef8542272535845e9408f14f151 | 84b42fa8b3d635d2 | 8cb42fa8b35635d2 |
| pair22_a.jpg / pair22_b.jpg | 2 | yes | shiprsimagenet__100001432_bmp.rf.017d2dffeca482a4f94c48d14bc907f5 | military_ships__100001432_bmp.rf.a0479d63bc75fb5f03749b0bef11dd9c | ab5295af62594cc3 | ab5295af62594d83 |
| pair23_a.jpg / pair23_b.jpg | 2 | yes | shiprsimagenet__100001554_bmp.rf.80247df448d9ffc12da827c75f4dc44e | military_ships__100001554_bmp.rf.f8148e0e3ed82975e0dc42caa73731c0 | 99ece6d266d0a4c3 | 99ece6d267d0a4c2 |
| pair24_a.jpg / pair24_b.jpg | 2 | yes | shiprsimagenet__100001477_bmp.rf.04968f18470c4f2f51cfb4784dee0011 | military_ships__100001477_bmp.rf.c6d5f50f000407572eef32fc6149031e | d2dedcd06f232710 | d2dfdcd06f032710 |
