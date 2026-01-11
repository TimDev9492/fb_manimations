from manim.manimlib import *

import numpy as np

from manim_slides import Slide

class PriceComputation(Slide):
    def setup(self):
        self.T = 0.5
        self.n = 3
        self.delta_t = self.T / self.n
        self.riskfree_rate = 0.06
        self.asset_volatility = 0.2
        self.asset_price = 100
        self.strike_price = 95
        self.tilt = 0
        self.discrete_rate_factor = np.exp(self.riskfree_rate * self.delta_t)

        self.u = np.exp(self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)
        self.d = np.exp(-self.asset_volatility * np.sqrt(self.delta_t) + self.tilt * self.asset_volatility**2 * self.delta_t)

        self.y_min = self.asset_price * self.d**self.n
        self.y_max = self.asset_price * self.u**self.n

        self.r = np.exp(self.riskfree_rate * self.delta_t) - 1
        self.p = (1 + self.r - self.d) / (self.u - self.d)
        self.K = 95

    def S(self, i, j):
        return self.asset_price * self.u**j * self.d**(i-j)

    def construct(self):
        self.setup()

        price_range = (np.log(self.y_min), np.log(self.y_max), (np.log(self.y_max) - np.log(self.y_min)) / self.n)
        t_range = (0, self.T, self.delta_t)

        t_axis = NumberLine(
            x_range=t_range,
            width=8,
            decimal_number_config={
                'font_size': 32,
                'num_decimal_places': 2,
            },
        )
        t_axis.add_numbers(font_size=32)
        t_axis.to_edge(DOWN)
        t_axis.to_edge(LEFT)
        self.play(ShowCreation(t_axis))

        price_axis = NumberLine(
            x_range=price_range,
            width=6,
            include_ticks=False,
            decimal_number_config={
                'font_size': 32,
                'num_decimal_places': 2,
            },
            line_to_number_direction=LEFT,
        )
        price_axis.rotate(90 * DEG, about_point=ORIGIN)

        prices = [
            self.S(i, j) for i in (self.n - 1, self.n) for j in range(i+1)
        ]
        prices.sort()

        ticks = VGroup()
        for price in prices:
            ticks.add(price_axis.get_tick(np.log(price), price_axis.tick_size))
        price_axis.add(ticks)

        for price, tick in zip(prices, ticks):
            label = DecimalNumber(price, font_size=32, num_decimal_places=2)
            always(label.next_to, tick, LEFT, SMALL_BUFF)
            price_axis.add(label)

        price_axis.add_numbers(prices)
        price_axis.to_edge(LEFT)
        # self.play(ShowCreation(price_axis))

        def c2p(t, S):
            t_aligned = t_axis.n2p(t)
            t_aligned[1] = price_axis.n2p(np.log(S))[1]
            return t_aligned

        NODE_COLOR = YELLOW_B
        EDGE_COLOR = YELLOW_E

        S_0 = Dot(c2p(0, self.asset_price), fill_color=NODE_COLOR)
        S_0.set_z_index(1)
        label = Tex(f'S_0 = {self.asset_price}', font_size=48)
        label.set_color(NODE_COLOR)
        label.next_to(S_0, RIGHT)

        self.play(GrowFromCenter(S_0), Write(label))

        self.next_slide()

        self.play(FadeOut(label))

        P_FONT_SIZE = 40
        P_COLOR = RED_A

        # add formulas
        # u_formula = Tex(r'u = e^{\sigma \sqrt{\Delta t}} \approx ' + str(round(self.u, 3)), font_size=48)
        # d_formula = Tex(r'd = e^{-\sigma \sqrt{\Delta t}} \approx ' + str(round(self.d, 3)), font_size=48)
        p_formula = Tex(r'p = \frac{1+r-d}{u-d}', font_size=32)
        S_formula = Tex(r'S_0 = ' + str(self.asset_price), font_size=32)
        K_formula = Tex(r'K = ' + str(self.K), font_size=32)
        values = VGroup(S_formula, K_formula).arrange(DOWN)
        # formulas = VGroup(u_formula, d_formula).arrange(DOWN)
        p_formula.next_to(c2p(0, self.y_max), RIGHT)
        values.next_to(p_formula, RIGHT, MED_LARGE_BUFF)
        # self.play(Write(p_formula))

        S_layers: VGroup[VGroup[VGroup]] = VGroup()
        S_layers.add(VGroup(VGroup(S_0)))

        for layer in range(1, self.n+1):
            layer_group = VGroup()
            prev_layer = S_layers[layer-1]
            lower_child = Dot(c2p(layer * self.delta_t, self.S(layer, 0)), fill_color=NODE_COLOR)
            lower_child.set_z_index(layer)
            layer_group.add(VGroup(lower_child))

            for up_moves in range(len(prev_layer)):
                node_group = prev_layer[up_moves]
                node = node_group[0]
                u_pos = c2p(layer * self.delta_t, self.S(layer, up_moves+1))
                d_pos = c2p(layer * self.delta_t, self.S(layer, up_moves))
                d_edge = DashedLine(node.get_center(), d_pos)
                d_edge.set_stroke(color=EDGE_COLOR, width=2)
                u_edge = DashedLine(node.get_center(), u_pos)
                u_edge.set_stroke(color=EDGE_COLOR, width=2)

                node_group.add(d_edge, u_edge)

                upper_child = Dot(u_pos, fill_color=NODE_COLOR)
                upper_child.set_z_index(layer)
                layer_group.add(VGroup(upper_child))

                if layer == 1:
                    u_label = Tex(f'\\times u', font_size=32)
                    d_label = Tex(f'\\times d', font_size=32)

                    u_label.next_to(u_edge, UP)
                    d_label.next_to(d_edge, DOWN)

                    # self.play(Write(d_label), ShowCreation(d_edge), GrowFromCenter(lower_child))
                    # self.next_slide()
                    # self.play(Write(u_label), ShowCreation(u_edge), GrowFromCenter(upper_child))
                    # self.next_slide()
                    # self.play(FadeOut(u_label), FadeOut(d_label))
                else:
                    pass
                    # if up_moves == 0:
                    #     self.play(ShowCreation(d_edge), ShowCreation(u_edge), GrowFromCenter(lower_child), GrowFromCenter(upper_child))
                    # else:
                    #     self.play(ShowCreation(d_edge), ShowCreation(u_edge), GrowFromCenter(upper_child))

            S_layers.add(layer_group)
            # if layer > 1:
            #     self.next_slide()

        self.play(ShowCreation(S_layers), Write(p_formula), Write(values))

        self.next_slide()

        display_layer = len(S_layers) - 1
        layer_nodes = [node_group[0] for node_group in S_layers[display_layer]]

        node_prices = [round(self.S(display_layer, j), 2) for j in range(len(layer_nodes))]
        node_price_labels = [Tex(f'{node_price}\\$', font_size=32) for node_price in node_prices]
        node_P_label_rhs = [f'({node_prices[j]} - {self.K})^+' for j in range(len(layer_nodes))]
        node_P_labels = [Tex(f'P_{display_layer}({j})', '=', f'{node_P_label_rhs[j]}') for j in range(len(layer_nodes))]
        for price_label, P_label in zip(node_price_labels, node_P_labels):
            price_label.set_color(NODE_COLOR)
            P_label.set_color(P_COLOR)

        for node, price_label, P_label in zip(layer_nodes, node_price_labels, node_P_labels):
            price_label.next_to(node, UP)
            P_label.next_to(node, RIGHT)
        
        self.play(*[Write(label) for label in node_price_labels])

        self.next_slide()

        FORMULA_FONT_SIZE=32
        FORMULA_COLOR = TEAL_A
        terminal_price_formula = Tex('P_n(j) = (S(n,j) - K)^+', font_size=FORMULA_FONT_SIZE)
        terminal_price_formula.set_color(FORMULA_COLOR)
        iterative_price_formula = Tex(r'P_{k-1}(j) = \frac{p P_k(j+1) + (1-p) P_k(j)}{1+r}', font_size=FORMULA_FONT_SIZE)
        iterative_price_formula.set_color(FORMULA_COLOR)
        final_price_formula = Tex(r'\pi(H) = P_0(0)', font_size=FORMULA_FONT_SIZE)
        final_price_formula.set_color(FORMULA_COLOR)

        price_algorithm = VGroup(terminal_price_formula, iterative_price_formula, final_price_formula).arrange(DOWN)
        price_algorithm.next_to(S_layers, RIGHT)

        self.play(Write(price_algorithm))

        self.next_slide()

        self.play(FadeOut(price_algorithm))

        self.play(*[Write(label) for label in node_P_labels])
        self.play(*[FadeOut(label) for label in node_price_labels])

        self.next_slide()

        leaf_P_values = [round(max(node_prices[j] - self.K, 0), 2) for j in range(len(layer_nodes))]
        node_P_labels_computed_rhs = [str(num) for num in leaf_P_values]
        node_P_labels_computed = [Tex(
            # f'P_{display_layer}({j})=',
            f'{node_P_labels_computed_rhs[j]}',
            font_size=P_FONT_SIZE
        ) for j in range(len(layer_nodes))]
        for label in node_P_labels_computed:
            label.set_color(P_COLOR)
        for node, P_comp in zip(layer_nodes, node_P_labels_computed):
            P_comp.next_to(node, UP)

        self.play(*[
            TransformMatchingTex(
                node_P_labels[i],
                node_P_labels_computed[i],
                key_map={
                    old: new for old, new in zip(node_P_label_rhs, node_P_labels_computed_rhs)
                }
            )
            for i in range(len(node_P_labels))
        ])

        self.play(Write(price_algorithm))

        self.next_slide()

        # Computation animation
        BRANCH_OPACITY = 0.2

        P_labels = node_P_labels_computed
        P_values = []
        P_values.append(leaf_P_values)

        num_layers = len(S_layers)
        for display_layer in range(num_layers - 1, 0, -1):
        # display_layer = num_layers - 1
            layer_P_values = []
            layer_nodes = [node_group[0] for node_group in S_layers[display_layer]]
            for current_ups in range(len(layer_nodes) - 2, -1, -1):
                # up_node = layer_nodes[current_ups+1]
                # down_node = layer_nodes[current_ups]

                parent_node_group = S_layers[display_layer-1][current_ups]
                parent_node = parent_node_group[0]
                up_branch = parent_node_group[2]
                down_branch = parent_node_group[1]

                up_P = P_values[num_layers - display_layer - 1][current_ups+1]
                down_P = P_values[num_layers - display_layer - 1][current_ups]

                up_label_old = P_labels[current_ups+1]
                prev_up_label = up_label_old.copy()
                prev_up_label.set_opacity(BRANCH_OPACITY)

                down_label_old = P_labels[current_ups]
                prev_down_label = down_label_old.copy()
                prev_down_label.set_opacity(BRANCH_OPACITY)

                up_label_new = Tex(r'p \times', f'{up_P}', font_size=P_FONT_SIZE)
                up_label_new.move_to(up_label_old.get_center())
                up_label_new.set_color(P_COLOR)
                down_label_new = Tex(r'(1-p) \times', f'{down_P}', font_size=P_FONT_SIZE)
                down_label_new.move_to(down_label_old.get_center())
                down_label_new.set_color(P_COLOR)

                plus_label = Tex('+', font_size=P_FONT_SIZE)
                plus_label.set_color(P_COLOR)
                plus_label.move_to(0.5 * (up_label_new.get_center() + down_label_new.get_center()))

                self.play(
                    up_branch.animate.set_opacity(BRANCH_OPACITY),
                    down_branch.animate.set_opacity(BRANCH_OPACITY),
                    TransformMatchingTex(
                        up_label_old,
                        up_label_new
                    ),
                    TransformMatchingTex(
                        down_label_old,
                        down_label_new
                    ),
                    run_time=0.8
                )

                self.play(Write(plus_label), run_time=0.5)

                self.next_slide()

                parent_P_bar = self.p * up_P + (1 - self.p) * down_P
                parent_P_bar_rounded = round(parent_P_bar, 2) if parent_P_bar != 0 else 0
                parent_P_value = round(parent_P_bar / (1 + self.r), 2)
                layer_P_values.append(parent_P_value if parent_P_value != 0 else 0)

                P_bar = Tex(str(parent_P_bar_rounded), font_size=P_FONT_SIZE)
                P_bar.set_color(P_COLOR)
                P_bar.move_to(plus_label.get_center())

                self.play(ReplacementTransform(VGroup(up_label_new, plus_label, down_label_new), P_bar), FadeIn(prev_up_label), FadeIn(prev_down_label), run_time=0.8)

                P_value_frac = Tex(*([str(parent_P_bar_rounded), r'\times (1+r)^{-1}'] if parent_P_bar_rounded != 0 else ['0']), font_size=P_FONT_SIZE)
                P_value_frac.set_color(P_COLOR)
                P_value_frac.next_to(parent_node, UP)
                if display_layer == 1:
                    P_value_frac.shift(RIGHT)

                self.play(TransformMatchingTex(P_bar, P_value_frac), run_time=1)

                self.next_slide()

                P_value = Tex(f'P_{display_layer-1}({current_ups})=' + str(parent_P_value if parent_P_value != 0 else 0), font_size=P_FONT_SIZE)
                P_value.set_color(P_COLOR)
                P_value.move_to(P_value_frac.get_center())

                self.play(ReplacementTransform(P_value_frac, P_value), run_time=0.5)
                if current_ups != 0:
                    self.play(prev_down_label.animate.set_opacity(1))
                P_labels[current_ups] = prev_down_label

                P_labels[current_ups+1] = P_value

                self.next_slide()


            layer_P_values.reverse()
            P_values.append(layer_P_values)
            P_labels.pop(0)

            # Remove the P_k notation
            raw_P_labels = [
                Tex(str(p_val), font_size=P_FONT_SIZE) for p_val in layer_P_values
            ] if display_layer > 1 else [Tex(r'\pi(H)', '=' + str(p_val), font_size=P_FONT_SIZE) for p_val in layer_P_values]
            for raw_label, p_label in zip(raw_P_labels, P_labels):
                raw_label.set_color(P_COLOR)
                raw_label.move_to(p_label.get_center())
            self.play(*[TransformMatchingTex(
                with_p,
                without_p,
                key_map={
                    'P_0(0)': r'\pi(H)'
                }
            ) for with_p, without_p in zip(P_labels, raw_P_labels)], run_time=0.5)
            P_labels = raw_P_labels

            self.next_slide()
