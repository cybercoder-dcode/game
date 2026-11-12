import pygame

_label_font = None  # created lazily, after pygame.init()


def compute_layout(genome, config, panel_w=320, panel_h=600, min_r=4, max_r=14,
                   input_labels=None, output_labels=None):
    """
    Build a static layout dict for a FeedForwardNetwork.
    Returns dict with keys:
        pos         : {node_id: (x, y)}
        panel_w, panel_h
        layers      : list of lists of node_ids (input layer first, then topological order)
        input_nodes : list of input node ids
        output_nodes: list of output node ids
        links       : list of (src, dst, weight) for enabled connections
        labels      : {node_id: str}  # human-readable labels for I/O nodes
        r, max_w    : node radius and max |weight| used for line thickness
    """
    # --- 1. Topological layers (same as neat.nn.feed_forward.layers) ---
    from neat.nn.feed_forward import feed_forward_layers
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    raw_layers = feed_forward_layers(input_keys, output_keys, genome.connections)

    # feed_forward_layers leaves the input nodes out, and its layers can hold
    # sets of node ids depending on the neat-python version, so flatten them
    # to plain ints. The input layer goes first so it gets drawn on the left.
    def _flatten_layer(layer):
        out = []
        for node in layer:
            if isinstance(node, (set, frozenset)):
                out.extend(int(n) for n in node)
            else:
                out.append(int(node))
        return out

    layers = [list(input_keys)] + [_flatten_layer(layer) for layer in raw_layers]
    layers = [layer for layer in layers if layer]

    # --- 2. Node positions ---
    n_layers = len(layers)
    margin_x = 60   # room on the left/right for labels
    margin_y = 40
    usable_w = panel_w - 2 * margin_x
    dx = usable_w / max(1, n_layers - 1) if n_layers > 1 else 0

    # shrink vertical spacing if the tallest layer would not fit the panel
    max_nodes = max(len(layer) for layer in layers)
    spacing = 40
    if max_nodes > 1:
        spacing = min(40, (panel_h - 2 * margin_y) / (max_nodes - 1))

    pos_map = {}
    for li, layer in enumerate(layers):
        x = margin_x + li * dx
        layer_h = (len(layer) - 1) * spacing
        start_y = (panel_h - layer_h) / 2
        for ni, node in enumerate(layer):
            y = start_y + ni * spacing
            pos_map[node] = (x, y)

    # --- 3. Connection list (only enabled) ---
    conn_list = []
    for cg in genome.connections.values():
        if cg.enabled:
            conn_list.append((cg.key[0], cg.key[1], cg.weight))

    # --- 4. Labels for I/O nodes ---
    labels = {}
    if input_labels:
        for nid, txt in zip(input_keys, input_labels):
            labels[nid] = txt
    if output_labels:
        for nid, txt in zip(output_keys, output_labels):
            labels[nid] = txt

    # avoid a divide-by-zero in draw_network when every connection weight is 0
    max_w = max((abs(w) for _, _, w in conn_list), default=1.0) or 1.0

    hidden_nodes = [n for layer in layers[1:] for n in layer if n not in output_keys]

    return {
        "pos": pos_map,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "layers": layers,
        "input_nodes": input_keys,
        "output_nodes": output_keys,
        "hidden_nodes": hidden_nodes,
        "links": conn_list,          # <-- draw_network expects "links"
        "labels": labels,
        "r": min_r,                  # <-- draw_network expects "r"
        "max_w": max_w,              # <-- draw_network expects "max_w"
    }


def get_activations(net, inputs):
    vals = {i: v for i, v in zip(net.input_nodes, inputs)}
    for node, act, agg, bias, response, links in net.node_evals:
        s = agg([vals[i] * w for i, w in links])
        vals[node] = act(bias + response * s)
    return vals


def _color(a):
    t = max(-1.0, min(1.0, a)) * 0.5 + 0.5
    return (int(30 + 180 * t), int(60 + 60 * (1 - abs(2 * t - 1))), int(230 - 180 * t))


def _get_label_font():
    global _label_font
    if _label_font is None:
        _label_font = pygame.font.SysFont(None, 18)
    return _label_font


def draw_network(surf, layout, vals, pos=(0, 0)):
    pos_map, r = layout["pos"], layout["r"]
    input_set = set(layout["input_nodes"])
    output_set = set(layout["output_nodes"])
    font = _get_label_font()

    panel = pygame.Surface((layout["panel_w"], layout["panel_h"]), pygame.SRCALPHA)
    panel.fill((10, 10, 20, 220))

    # connections first so the nodes sit on top of them
    for src, dst, w in layout["links"]:
        if src not in pos_map or dst not in pos_map:
            continue
        c = (120, 255, 120, 255) if w >= 0 else (255, 120, 120, 255)
        pygame.draw.line(panel, c, pos_map[src], pos_map[dst], max(1, int(5 * abs(w) / layout["max_w"])))

    # nodes, coloured by activation
    for n, (x, y) in pos_map.items():
        pygame.draw.circle(panel, _color(vals.get(n, 0.0)), (int(x), int(y)), r)
        pygame.draw.circle(panel, (255, 255, 255), (int(x), int(y)), r, 1)

    # labels: inputs to the left of their nodes, outputs to the right
    # (hidden nodes get a short "H" tag), so text never overlaps the wires
    for n, (x, y) in pos_map.items():
        if n in layout["labels"]:
            txt = layout["labels"][n]
        elif n in layout["hidden_nodes"]:
            txt = "H"
        else:
            continue
        label = font.render(txt, True, (235, 235, 235))
        if n in input_set:
            panel.blit(label, (x - r - 6 - label.get_width(), y - label.get_height() // 2))
        elif n in output_set:
            panel.blit(label, (x + r + 6, y - label.get_height() // 2))
        else:
            panel.blit(label, (x - label.get_width() // 2, y - r - label.get_height()))

    # title + hidden-layer count, so it is obvious what the network contains
    title = font.render(
        f"Champion network: {len(layout['hidden_nodes'])} hidden node(s), "
        f"{len(layout['links'])} connection(s)", True, (255, 255, 255))
    panel.blit(title, (10, 8))

    surf.blit(panel, pos)